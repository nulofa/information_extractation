import numpy as np
import requests
import re
def getHtmlText(url):
    try:
        r = requests.get(url, timeout = 30)
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r.text;
    except:
        return "产生异常"

sam_d = np.load('sam_details_11_12.npy', allow_pickle=True).item()

cnt = 1
train_X = []
train_Y = []
gse_design = {}
sam_rep = {}
for sc in sam_d:   #(gsm, [title], [characs])
    gsm = sam_d[sc][0]

    characs = sam_d[sc][2][0]
    characs = characs.split('<br>')
    if np.sum(['genotype' in ch for ch in characs]) == 0:
        print('has no genotype..',characs)
        continue
    else:
        gt = characs[['genotype' in ch for ch in characs].index(True)].split(':')[1]
        gt = gt.strip()
    print(sam_d[sc])
    url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + gsm
    gsmcont = getHtmlText(url)
    # print(gsmcont)
    title = sam_d[sc][1]
    gse = re.findall('/geo/query/acc.cgi\?acc=(GSE\d+)"', gsmcont)[0]

    if gse+title[0][:-1] in sam_rep:
        continue
    else:
        sam_rep[gse+title[0][:-1]] = 1

    # print(gsm)

    if gse in gse_design:
        design = gse_design[gse]
    else:
        url2 = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + gse
        gsecont = getHtmlText(url2)
        # print(gsecont)
        design = re.findall('<tr valign="top"><td nowrap>Overall design</td>\n<td style="text-align: justify">(.*)<br></td>',gsecont)
        # print(design)
        gse_design[gse] = design

    treatment_protocol = re.findall('<tr valign="top"><td nowrap>Treatment protocol</td>\n<td style="text-align: justify">(.*)<br></td>', gsmcont)
    # print(title,treatment_protocol)
    if len(treatment_protocol) <1:
        treatment_protocol = ['']
    x = design[0] + '---' + title[0] + '---' + treatment_protocol[0]+'---'+gsm
    y = gt

    train_X.append(x)
    train_Y.append(y)
    # if cnt > 1000:
    #     break
    cnt +=1
    print(cnt)
np.save('X_gt', train_X)
np.save('Y_gt', train_Y)

