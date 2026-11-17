"""True sample-index SVG: full gain series; disclosed left-channel edge zoom."""
import csv,html
from pathlib import Path
P=Path(__file__).resolve().parent;rows=list(csv.DictReader((P/'samples.csv').open()))
def T(x,y,s,size=15,c='#edf0e9'):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{c}">{html.escape(s)}</text>'
def draw(mobile):
 W=375 if mobile else 1200;H=880 if mobile else 690;left=55;pw=W-80;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc"><title id="title">One event revised; unselected samples preserved</title><desc id="desc">Full one-second layer gain at 25 percent before and 50 percent after, with 160-sample fades inside the selected 800 to 2400 sample span. A separate left-channel zoom shows the actual original and exported mix at the start of the edit. Every sample outside the selected span is retained in both channels. Synthetic supplied layer, not separator output.</desc><rect width="100%" height="100%" fill="#161a18"/><g font-family="Arial, Helvetica, sans-serif">',T(22,34,'STEMSCENE / REVERSIBLE AUDIO',13,'#b6cba0')]
 if mobile:out +=[T(22,70,'One event revised.',26),T(22,102,'Every other sample kept.',25)]
 else:out +=[T(22,76,'One event revised. Every other sample kept.',36)]
 top=180 if mobile else 155;gh=110 if mobile else 125
 out +=[T(22,140 if mobile else 116,'01 / LAYER GAIN · FULL 1 SECOND',13,'#b6cba0')]
 for val in [0,.5,1]:
  y=top+gh*(1-val);out.append(f'<line x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}" stroke="#424b44"/>');out.append(T(22,y+4,f'{val:g}',12,'#bdc6bc'))
 for i,key in enumerate(['gain_before','gain_after']):
  pts=' '.join(f'{left+float(r["seconds"])*pw:.9f},{top+gh*(1-float(r[key])):.9f}'for r in rows);out.append(f'<polyline data-series="{key}" fill="none" stroke="{["#b7b2da","#c9dcad"][i]}" stroke-width="2"'+(' stroke-dasharray="5 4"'if i==0 else'')+f' points="{pts}"/>')
 for t in [0,.1,.3,1]:out.append(T(left+t*pw-4,top+gh+22,f'{t:g}',12,'#bdc6bc'))
 labely=top+gh+55;out +=[T(22,labely,'25% before / 50% after',16),T(22,labely+25,'Same event ID; replacement, not stacking.',13,'#bdc6bc')]
 if not mobile:out +=[T(625,labely,'Selected samples [800, 2400) at 8 kHz',17),T(625,labely+25,'0.10–0.30 s · 160-sample inside-span fades',14,'#bdc6bc')]
 else:out +=[T(22,labely+51,'[800, 2400) samples · 0.10–0.30 s',14,'#bdc6bc')]
 ztop=492 if mobile else 460;zh=145 if mobile else 135;lo=.09;hi=.14
 out.append(T(22,ztop-35,'02 / LEFT CHANNEL · 0.09–0.14 s ZOOM',12 if mobile else 13,'#b6cba0'))
 for v in [-.2,0,.2]:
  y=ztop+zh*(.2-v)/.4;out.append(f'<line x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}" stroke="#424b44"/>');out.append(T(12,y+4,f'{v:g}',12,'#bdc6bc'))
 xx=left+(.1-lo)/(hi-lo)*pw;out.append(f'<line x1="{xx}" y1="{ztop-10}" x2="{xx}" y2="{ztop+zh}" stroke="#b6cba0" stroke-dasharray="3 4"/>')
 for i,key in enumerate(['original_left','mix_left']):
  pts=' '.join(f'{left+(float(r["seconds"])-lo)/(hi-lo)*pw:.9f},{ztop+zh*(.2-float(r[key]))/.4:.9f}'for r in rows if lo<=float(r['seconds'])<=hi);out.append(f'<polyline data-series="{key}" fill="none" stroke="{["#b7b2da","#c9dcad"][i]}" stroke-width="{1.6 if i==0 else 2}"'+(' stroke-dasharray="4 3"'if i==0 else'')+f' points="{pts}"/>')
 for t in [.09,.1,.14]:out.append(T(left+(t-lo)/(hi-lo)*pw-10,ztop+zh+23,f'{t:g}',12,'#bdc6bc'))
 out +=[T(22,ztop+zh+53,'Original (dashed) / exported mix (solid)',13,'#bdc6bc')]
 if mobile:
  for y,t in [(742,'Linear sample amplitude, not loudness.'),(773,'Both channels: unchanged outside event.'),(805,'Supplied layer = 0.6 × original.'),(835,'Synthetic editing fixture. No inference.')]:out.append(T(22,y,t,13,'#bdc6bc'))
 else:out +=[T(22,675,'Synthetic supplied layer = 0.6 × original. Linear sample amplitude, not loudness. Both channels checked; no separator inference.',15,'#bdc6bc')]
 out.append('</g></svg>');return ''.join(out)
for mobile in [False,True]:(P/('event-mobile.svg'if mobile else'event.svg')).write_text(draw(mobile))
