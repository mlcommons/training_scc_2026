# NVIDIA's scripts/download_dataset.py:transform_row, run INSIDE the NGC container
# so numpy/pandas versions match those used to produce the official .npy files.
import numpy as np, pandas as pd, hashlib, os
O='/wk/data/dataset/OFFICIAL_scrolls/data'; D='/wk/data/dataset/nemo_npy'
os.makedirs(D, exist_ok=True)
exp={'train':(192034874,'c260a8d97b5698dc8c1d9be5b1c765e1e8f654a07cf3f1bba8de2ab1d86fdeee'),
     'validation':(8516614,'0cef3600453e44cdee45de214b158d55997fcdd251e244b070a156beb38bb3bc')}
print("numpy",np.__version__,"pandas",pd.__version__)
for s in ('train','validation'):
    df=pd.read_parquet(f'{O}/{s}-00000-of-00001.parquet')
    rows=df.apply(lambda r:{"input_ids":r["input_ids"],
                            "loss_mask":[int(x!=-100) for x in r["labels"]],
                            "seq_start_id":[0]}, axis=1).tolist()
    out=f'{D}/{s}.npy'; np.save(out, rows)
    n=os.path.getsize(out); h=hashlib.sha256(open(out,'rb').read()).hexdigest()
    en,eh=exp[s]
    print(f'{s}.npy size={n} expected={en} {"SIZE-OK" if n==en else "SIZE-DIFF("+str(n-en)+")"}  sha256 {"MATCH" if h==eh else "differs"}')
