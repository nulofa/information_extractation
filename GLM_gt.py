import numpy as np
X = np.load('X_gt_1000.npy')
Y = np.load('Y_gt_1000.npy')

def history(x, i):
    return (x[i-2:i],x, i)

def creat_feature(hi, tag):
    sentence = hi[1]
    pre_2 = hi[0]
    pos = hi[2]
    word_count = len(sentence)
    wi = sentence[pos]
    ti = tag[pos]

    f = []
    design = ''.join(sentence).split('---')[0]
    ti_in_design = np.sum(tag[:len(design)]) > 0 # tag出现在design里面
    title = ''.join(sentence).split('---')[1]
    wi_in_title = wi in title

    ind_of_delimiter = {}
    for i, v in enumerate(sentence):
        if v == '---':
            if v not in ind_of_delimiter:
                ind_of_delimiter[v] = [i]
            else:
                ind_of_delimiter[v].append(i)
    ind_of_delimiter = ind_of_delimiter['---']
    pos_in_title = pos > ind_of_delimiter[0] and pos<ind_of_delimiter[1]
    f.append('01'+ str(wi_in_title) + str(ti)+str(pos_in_title))
    f.append('02'+str(ti_in_design)+str(ti)+str('col' in wi.lower())+str(pos_in_title))
    return f


def dp_tag(x, v, xi, prob_tag,tagscore):
    if xi in tagscore:
        return tagscore[xi]
    if xi==0:
        prob_tag[xi] = 1
        f1 = get_fearture(x, prob_tag)
        s1 = np.dot(f1,v)
        prob_tag[xi] = 0
        f2 = get_fearture(x, prob_tag)
        s2 = np.dot(f2,v)
        tscore = s2
        if s1>tscore:
            tscore = s1
            prob_tag[xi] = 1
        tagscore[xi] = tscore
        # print(tagscore)
        return tscore

    res0 = dp_tag(x,v, xi-1, prob_tag, tagscore)
    prob_tag[xi] = 1
    f1 = get_fearture(x, prob_tag)
    prob_tag[xi] = 0
    f2 = get_fearture(x, prob_tag)

    res1 = res0 + np.dot(f1, v)
    res2 = res0 + np.dot(f2,v)
    if res1 > res2:
        prob_tag[xi] = 1
        tagscore[xi] = res1
        return res1
    else:
        prob_tag[xi] = 0
        tagscore[(xi)] = res2
        return res2

def get_fearture(x, tag):
    feature = np.zeros(10)
    for i in range(2, len(x)):
        # print(x)
        fi = np.zeros(10)
        hi = history(x, i)
        fs  = creat_feature(hi, tag)    #一个sentence的包含所有Feature
        # print(fs)
        for f in fs:
            # print(f)
            if f in feature2ind:    # f是否标为1，f in Feature2ind 说明满足自定义的特征要求
                fi[feature2ind[f]] = 1
        feature += fi
    return feature



#f01: tag==1出现在title里面
#f02: tag=1出现在design里面，且没出现再title，但在title出现col
feature2ind = {'01True1True':0, '02True0TrueTrue':1}
vec = np.zeros(10)
for it in range(1):
    i4x = 0
    for x in X[:100]:

        x = x.replace(',', ' ')  # 统一分隔符， 将，转为空格
        x = x.replace('  ', ' ')  # 原本的，空格由于上步变为两个空格，转为一个空格
        x = x.replace('(', '')  # 去除括号
        x = x.replace(')', '')
        x = x.replace('---', ' --- ')  # 使---单独一行
        x = '* * ' + x

        gt = Y[i4x].lower()
        i4x += 1


        if not gt in x.lower():
            if 'wild' in gt or 'wt' in gt:
                ogt = gt
                gt = ['wt', 'wild type', 'wild-type', 'wild_type','col']

        if len(gt[0])>1:    #多别名
            if np.sum([gti in x.lower() for gti in gt]) == 0: #多个别名也没在X中
                continue
            else:
                gt = gt[[gti in x.lower() for gti in gt].index(True)]
        else:   #单一名称
            if gt not in x.lower():
                # print(gt)
                # print(x)
                continue

        x = x.lower().split(' ')
        gt = gt.split(' ')
        if np.sum(['mutant' in g for g in gt]) > 0:
            gt.pop(['mutant' in g for g in gt].index(True))
        tag = [0 for i in range(len(x))]
        #--出现genotype的index
        g0 = gt[0]
        get_ind = {}
        for i, v in enumerate([g0 in xx for xx in x]):
            if v:
                if v not in get_ind:
                    get_ind[v] = [i]
                else:
                    get_ind[v].append(i)
        #---
        print(i4x)
        i4g0s = get_ind[True]
        for i4g0 in i4g0s:
            # print(gt[:],x[i4g0:i4g0+len(gt)])
            if ''.join(gt[:]) in ''.join(x[i4g0:i4g0+len(gt)]):
                tag[i4g0:i4g0+len(gt)] = [1 for i in range(len(gt))]
            # print(x,tag)

        feature = get_fearture(x, tag)
        # print(feature)
        # print(x)

        #a varient of perceptron
        most_prob_tag = [0 for i in range(len(x))]
        tagscore = {}


        dp_tag(x,vec,len(x)-1,most_prob_tag,tagscore)


        if np.sum(vec) > 0 :
            print(most_prob_tag)
            print(tag, len(tag))
            print('-------')
            # print('tags=', tagscore)
            # break

        pred_f = get_fearture(x, most_prob_tag)
        # assert 1==0
        if np.sum(np.array(most_prob_tag) == np.array(tag)) < len(tag):
            vec += feature - pred_f
            # print('vec = ',vec)
        # print(i4x)

# test
print('test-------')
for x in X[900:910]:
    x = x.replace(',', ' ')  # 统一分隔符， 将，转为空格
    x = x.replace('  ', ' ')  # 原本的，空格由于上步变为两个空格，转为一个空格
    x = x.replace('(', '')  # 去除括号
    x = x.replace(')', '')
    x = x.replace('---', ' --- ')  # 使---单独一行
    x = '* * ' + x
    x = x.lower().split(' ')

    y_tag = [0 for i in range(len(x))]
    tagscore = {}
    dp_tag(x, vec, len(x)-1,y_tag,tagscore)
    print(y_tag)
    print(x)
    inds = np.array(y_tag)==1
    print(np.array(x)[inds])
    print('----')