import lmdb
from tqdm.auto import tqdm

n_bytes = 2**40
env_comb = lmdb.open("processed/trainvalid/lmdb_videos/trainvalid", map_size=n_bytes)




env_train = lmdb.open("processed/train/lmdb_videos/train")
env_valid = lmdb.open("processed/validation/lmdb_videos/valid")

txn_comb = env_comb.begin(write=True)


with env_train.begin(write=False) as txn:
   for key, value in tqdm(txn.cursor()):
       txn_comb.put(key=key, value=value)

txn_comb.commit()  
txn_comb = env_comb.begin(write=True)     

with env_valid.begin(write=False) as txn:
   for key, value in tqdm(txn.cursor()):
       txn_comb.put(key=key, value=value)
txn_comb.commit()  

env_comb.close()