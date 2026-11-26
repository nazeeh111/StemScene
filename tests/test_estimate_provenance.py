"""An estimate's saved provenance can change without changing its audio."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from stemscene.project import Anchor, Project


class EstimateProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.samples = np.zeros((800, 1))
        source = self.root / 'source.wav'
        sf.write(source, self.samples, 8000, subtype='DOUBLE')
        self.project = Project.create(source, self.root / 'project')
        self.identifier = self.project.add_estimates({'silence': self.samples}, metadata={'run': 'old'})['silence']

    def test_new_provenance_reuses_asset_and_identity_and_undo_restores_it(self):
        project = self.project
        event = project.edit(self.identifier, 0, .05, .25, 0)
        before = project.revision
        row = project._manifest['layers'][0].copy()
        returned = project.add_estimates({'silence': self.samples}, metadata={'run': 'new'})
        self.assertEqual(returned['silence'], self.identifier)
        self.assertEqual(project.revision, before + 1)
        saved = Project.load(project.root)
        self.assertEqual(saved.layers[0].metadata['run'], 'new')
        self.assertEqual(saved._manifest['layers'][0]['path'], row['path'])
        self.assertEqual(saved._manifest['layers'][0]['sha256'], row['sha256'])
        self.assertEqual(len(list((project.root / 'layers').glob('*.npy'))), 1)
        self.assertEqual(saved.edits[0]['id'], event)
        np.testing.assert_array_equal(saved.render(), self.samples)
        self.assertTrue(saved.undo())
        self.assertEqual(saved.layers[0].metadata['run'], 'old')
        self.assertEqual(saved.edits[0]['id'], event)

    def test_explicit_anchors_persist_and_omitted_or_identical_state_is_noop(self):
        project = self.project
        anchor = Anchor(0, .05, False)
        project.add_estimates({'silence': self.samples}, anchors={'silence': [anchor]})
        self.assertEqual(Project.load(project.root).layers[0].anchors, (anchor,))
        self.assertEqual(project.layers[0].metadata['run'], 'old')
        revision = project.revision
        document = (project.root / 'project.json').read_bytes()
        project.add_estimates({'silence': self.samples})
        project.add_estimates({'silence': self.samples}, anchors={'silence': [anchor]}, metadata={'run': 'old'})
        self.assertEqual(project.revision, revision)
        self.assertEqual((project.root / 'project.json').read_bytes(), document)
        project.add_estimates({'silence': self.samples}, anchors={'silence': []})
        self.assertEqual(Project.load(project.root).layers[0].anchors, ())
        self.assertEqual(project.revision, revision + 1)
        project.add_estimates({'silence': self.samples}, metadata={'flag': 1})
        revision = project.revision
        project.add_estimates({'silence': self.samples}, metadata={'flag': True})
        self.assertEqual(project.revision, revision + 1)
        self.assertIs(Project.load(project.root).layers[0].metadata['flag'], True)

    def test_locked_provenance_is_preserved_while_unlocked_neighbor_updates(self):
        project = self.project
        other = project.add_estimates({'other': self.samples}, metadata={'run': 'other-old'})['other']
        project.set_locked(self.identifier)
        locked = project._manifest['layers'][0].copy()
        project.add_estimates({'silence': self.samples, 'other': self.samples}, metadata={'run': 'new'})
        self.assertEqual(project._manifest['layers'][0], locked)
        self.assertEqual(project._find_row(other)['metadata']['run'], 'new')
        revision = project.revision
        project.add_estimates({'silence': self.samples}, metadata={'run': 'ignored-locked'})
        self.assertEqual(project.revision, revision)
        self.assertTrue(Project.load(project.root).layers[0].locked)

    def test_failed_metadata_commit_preserves_saved_state_and_allows_retry(self):
        project = self.project
        before = (project.root / 'project.json').read_bytes()
        revision = project.revision
        with patch('stemscene.project._write_json', side_effect=OSError('synthetic write failure')):
            with self.assertRaisesRegex(OSError, 'synthetic write failure'):
                project.add_estimates({'silence': self.samples}, metadata={'run': 'new'})
        self.assertEqual(project.revision, revision)
        self.assertEqual(project.layers[0].metadata['run'], 'old')
        self.assertEqual((project.root / 'project.json').read_bytes(), before)
        self.assertEqual(Project.load(project.root).layers[0].metadata['run'], 'old')
        self.assertEqual(len(list((project.root / 'layers').glob('*.npy'))), 1)
        project.add_estimates({'silence': self.samples}, metadata={'run': 'new'})
        self.assertEqual(Project.load(project.root).layers[0].metadata['run'], 'new')


if __name__ == '__main__':
    unittest.main()
