"""Local filesystem ownership and atomic no-replace publication."""
from contextlib import contextmanager
import ctypes
import errno
import os
from pathlib import Path
import stat
import sys
import tempfile

from .audio import AudioContractError

LOCK_FORMAT = b"stemscene-kernel-lock/1\n"


def _open_lock(path):
    flags = os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except FileNotFoundError:
        descriptor, temporary = tempfile.mkstemp(prefix=".kernel-lock-", dir=path.parent)
        try:
            os.write(descriptor, LOCK_FORMAT)
            os.fsync(descriptor)
            try:
                # An initialized inode appears atomically. A crash cannot leave
                # an ambiguous empty canonical marker from this protocol.
                os.link(temporary, path)
            except FileExistsError:
                os.close(descriptor)
                descriptor = None
                return os.open(path, flags)
            return descriptor
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise
        finally:
            os.unlink(temporary)


def _migrate_lock(path, fcntl):
    descriptor, temporary = tempfile.mkstemp(prefix=".kernel-lock-", dir=path.parent)
    try:
        os.write(descriptor, LOCK_FORMAT)
        os.fsync(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Publish already-initialized, already-locked ownership. A killed
        # migration leaves either the old dead marker or a complete new inode.
        os.replace(temporary, path)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


@contextmanager
def project_lock(root: Path, *, shared: bool = False):
    try:
        import fcntl
    except ImportError as exc:
        raise AudioContractError("Project writing requires macOS/Linux advisory file locking") from exc
    path = root / ".write.lock"
    descriptor = _open_lock(path)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 128:
            raise AudioContractError("Project lock must be an ordinary owned file; do not delete a live writer's lock")
        try:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AudioContractError("Project is being written; retry after the active operation finishes") from exc
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise AudioContractError("Lock ownership changed during admission; retry without deleting the lock")
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = os.read(descriptor, 129)
        if content != LOCK_FORMAT:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AudioContractError("Legacy lock migration is busy; retry without deleting it") from exc
            current = os.stat(path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise AudioContractError("Lock ownership changed during migration; retry")
            # Old releases used only a PID marker. Existing marker existence
            # prevents another old writer starting during this one-time migration.
            if info.st_nlink != 1:
                raise AudioContractError("Legacy lock has unverified links; preserve it for manual recovery")
            try:
                pid = int(content)
                if not 1 <= pid <= 2147483647 or str(pid).encode("ascii") != content:
                    raise ValueError
            except ValueError as exc:
                raise AudioContractError("Unrecognized legacy lock; preserve it and close old StemScene versions before manual recovery") from exc
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise AudioContractError("Legacy writer identity cannot be checked; close the old version before recovery") from exc
            else:
                raise AudioContractError("Legacy writer may still be alive (or its PID reused); close the old version before recovery")
        if content != LOCK_FORMAT:
            previous = descriptor
            descriptor = _migrate_lock(path, fcntl)
            os.close(previous)
        if shared:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)
    # Never unlink this inode: a waiting writer must lock this same file.


def publish_directory(staged: Path, destination: Path) -> None:
    """Kernel no-replace rename, including a concurrently created empty folder."""
    library = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staged), os.fsencode(destination)
    if sys.platform == "darwin":
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source, target, 0x00000004)  # Darwin RENAME_EXCL.
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE.
    else:
        raise AudioContractError("Atomic no-replace project publication requires supported macOS/Linux storage")
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(code, os.strerror(code), str(destination))
        raise OSError(code, os.strerror(code), str(destination))
