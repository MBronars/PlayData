import imageio, numpy as np, json, sys
from PIL import Image, ImageDraw
D, ep, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
nseg = int(sys.argv[4]) if len(sys.argv) > 4 else 6
r = [json.loads(l) for l in open(f'{D}/log.jsonl')]; r = [x for x in r if x.get('ep') == ep][0]
F = [f for f in imageio.get_reader(f'{D}/video/episode_{ep:07d}.mp4')]
rows = []
for s in r['segments'][:nseg]:
    idx = np.linspace(s['start'], min(s['end'], len(F)) - 1, 8).astype(int)
    ims = []
    for i in idx:
        im = Image.fromarray(F[i]); ImageDraw.Draw(im).text((3, 3), f"{i} {s['primitive']} {s['arm']} {s['success']}", fill=(255, 0, 0)); ims.append(np.array(im.resize((200, 150))))
    rows.append(np.concatenate(ims, 1))
Image.fromarray(np.concatenate(rows, 0)).save(out)
print([(s['primitive'], s['language'], s['success'], s.get('grasp')) for s in r['segments'][:nseg]])
