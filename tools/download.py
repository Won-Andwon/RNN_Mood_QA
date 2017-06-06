import os
from six.moves import urllib


def reporthook(downloaded_block, blocksize, totalsize):
  # 运算顺序防溢出
  percent = 100.0 * ((downloaded_block * blocksize) / totalsize)
  if percent > 100:
    print("???bigger than all???")
    percent = 100
  print("%.2f%%" % percent)


def download(directory, filename, url, report=True):
  if not os.path.exists(directory):
    print("Make a new directory")
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading...")
    if report:
      filepath, _ = urllib.request.urlretrieve(url, filepath, reporthook)
    else:
      filepath, _ = urllib.request.urlretrieve(url, filepath)
    
    print("Downloaded!")
    stat = os.stat(filepath)
  return filepath


# directory = r"D:\Data"
# filename = r"UNv1.0.en-zh.tar.gz.00"
# url = r"https://conferences.unite.un.org/UNCorpus/en/Download?file=UNv1.0.en-zh.tar.gz.00"
#
# download(directory, filename, url)