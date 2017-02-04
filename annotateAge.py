import cognitive_face as CF
import os
import glob
import time

with open(os.environ['HOME'] + '/COGFACE.txt', 'r') as f:
    KEY = f.readline().strip()
CF.Key.set(KEY)

already_seen = []
with open('./ageGenderAnnotations.csv', 'r') as inf:
    for line in inf:
        line = line.strip().split(',')
        already_seen.append(line[0])

annofile = './ageGenderAnnotations.csv'

flatten = lambda l: [item for sublist in l for item in sublist]

ages = {}
genders = {}

pre = "./overweightPictures/*."
files_grabbed = flatten([glob.glob(e) for e in [pre+'jpg', pre+'jpeg', pre+'png', pre+'gif']])

header_needed = False
if os.path.getsize(annofile) == 0:
    header_needed = True

outf = open(annofile, 'a')
if(header_needed):
    outf.write('id,age,gender\n')

for fp in files_grabbed:
    fnWithExtension = os.path.basename(fp)
    fn = os.path.splitext(fnWithExtension)[0]
    if (fn not in already_seen):
        try:
            resp = CF.face.detect(fp, attributes='age,gender')
            if(len(resp) > 0):
                age = resp[-1]['faceAttributes']['age']
                gender = resp[-1]['faceAttributes']['gender']
                outf.write(str(fn) + ',' + str(age) + ',' + gender + '\n')
            else:
                outf.write(str(fn) + ',NA,NA\n')
        except:
            print(fn + " failed")
            outf.write(str(fn) + ',NA,NA\n')
        time.sleep(3.1) # API rate limit: 20 requests per minute

outf.close()
