import math
from tqdm import tqdm
import requests
import time 
import os 
import sys
import pandas as pd 
import zipfile


def remove_li_downloaditou(paths):
    for i in paths:
        if os.path.exists(i):
            os.remove(i)
        else:
            print(f"The file {i} does not exist")

def unzip(path_zip,where_to_extract):
    print("Extracting data ...")
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(where_to_extract)
    remove_li_downloaditou([path_zip])
    print("Done extracting ...")
    return True

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def telechargi(id,dst_file,skip_if_exists=True):

    url=f"http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/{id}.zip"
    print("Downloading {} to {}.".format(url.split("/")[-1], "/".join(dst_file.split("\\")[-3:])))

    if skip_if_exists and os.path.isfile(dst_file):
        print("{} already exists. Skipped.".format(dst_file))
        return True

    dst_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir):
        print("Creating directory {}".format(dst_dir))
        os.makedirs(dst_dir)

    start=time.time()
    try:
        r=requests.get(url,stream=True)
        if r.ok:
            if int(r.headers['Content-Length'])>15000000000 : 
              print(convert_size(int(r.headers['Content-Length'])))
              return False
            num_bytes=0
            with open(dst_file,"wb") as f:
                pbar=tqdm(total=int(r.headers['Content-Length']),unit="bytes",desc=f"Downloading file ({convert_size(int(r.headers['Content-Length']))})")
                for chunk in r.iter_content(32768):
                    pbar.update(32768)
                    num_bytes += len(chunk)
                    f.write(chunk)
            pbar.close()
            mbytes = num_bytes / float(1000000)
            elapsed_time = time.time() - start
            speed = mbytes / elapsed_time
            print("Downloaded {:.2f}MB, speed {:.2f}MB/s. in {:.2f}seconds".format(
                mbytes, speed,elapsed_time))
            unzip(dst_file,os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shapenet", f"{id}"))
            return True
        else :
            print("Download request failed.")
            return False
    except:
        e = sys.exc_info()[0]
        print("Download request failed with exception {}.".format(e))
        return False



def downloader(id, skip_if_exists=True):
    scan_id=id.split("\\")[0]
    # print(scan_id)
    _pwd = os.path.dirname(os.path.abspath(__file__))
    dst_file = os.path.join(_pwd, "data", "shapenet", "{}.zip".format(scan_id))
    telechargi(scan_id, dst_file, skip_if_exists=skip_if_exists)

import random

csv=pd.read_csv("all.csv" ,dtype=str)
ids=csv["synsetId"]+"\\"+csv["subSynsetId"]+"\\"+csv["modelId"]
ids=list(set(list(ids)))

#downloader(random.choice(ids))
# downloader(random.choice(ids))
