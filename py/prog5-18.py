import numpy as np

# scikitlearn����A���t����w�K�ɕK�v�ȃ��W���[�����C���|�[�g
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# python�̃��X�g���Anumpy�z��ɕϊ�
X = np.array(docs2) 
y = np.array(labels)

# �P���f�[�^�ƃe�X�g�f�[�^�ɕ���
train_x, test_x, train_y, test_y, train_i, test_i \
    = train_test_split(docs2, labels, inds, \
    test_size=0.2)

# scikitlearn����ASVM�̃��W���[�����C���|�[�g	
from sklearn.svm import SVC

# SVM���w�K
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(train_x, train_y) 