# �v���O����2.6

import numpy as np
import pandas as pd
import random
# ROC�Ȑ��̕`��p�ɃC���|�[�g
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
# �����s��̉���(heatmap)�p�ɃC���|�[�g
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, \
confusion_matrix, classification_report, \
roc_curve, roc_auc_score
# ���x���G���R�[�f�B���O�p�ɃC���|�[�g
from sklearn.preprocessing import LabelEncoder
# �s�ύt�f�[�^���������C�u�����̃C���X�g�[��
# (Google Colab�ɃC���X�g�[���ς݂̂��̂̓o�[�W�������Â�)
get_ipython().system('pip3 install imbalanced-learn==0.7.0')
# �A���_�[�T���v�����O�p���C�u�������C���|�[�g
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
# �I�[�o�[�T���v�����O�p���C�u�������C���|�[�g
from imblearn.over_sampling import SMOTE,\
 ADASYN, RandomOverSampler

# �f�[�^�̏���
def prepare(test_count):
    #!kaggle datasets download -d ang3loliveira/malware-analysis-datasets-pe-section-headers
    get_ipython().system('unzip malware-analysis-datasets-pe-section-headers.zip')
    # �}���E�F�A����̕s�ύt�f�[�^���g�p
    df_train = pd.read_csv('pe_section_headers.csv')
    # ���ނɎg�p���������
    features = [c for c in df_train.columns.values[:4]]
    le = LabelEncoder()
    # �n�b�V���l�̕���������x���G���R�[�h����
    df_train['hash'] = le.fit_transform(df_train['hash'])
    X_train = df_train.loc[:,features].values
    y_train = df_train.loc[:,['malware']].values
    # ����A��������ꂼ��test_count���A�e�X�g�p�Ƃ���
    # �c��̃f�[�^���w�K�f�[�^�Ƃ���
    mal_ids = [i for i, e in enumerate(y_train) if e == 1]
    good_ids = [i for i, e in enumerate(y_train) if e == 0]
    random.seed(0)
    # �C���f�b�N�X���V���b�t�����A�f�[�^����בւ���
    random.shuffle(mal_ids) 
    random.shuffle(good_ids)
    X_test = X_train[mal_ids[:test_count] + good_ids[:test_count]] 
    y_test = y_train[mal_ids[:test_count] + good_ids[:test_count]]
    X = X_train[mal_ids[test_count:] + good_ids[test_count:]] 
    y = y_train[mal_ids[test_count:] + good_ids[test_count:]]
    y = y.ravel()
    y_test = y_test.ravel()
    return X, y, X_test, y_test, features

# ���T���v�����O(ENN, RUS, SMOTE, ROS, ADASYN��5���)
def sampling(sampling_type, X_train, y_train):
    print('\nSampling Type: %s' % sampling_type)
    if sampling_type == 'ENN':
      smp = EditedNearestNeighbours()
    elif sampling_type == 'RUS':
      smp = RandomUnderSampler()
    elif sampling_type == 'ROS':
      smp = RandomOverSampler()
    elif sampling_type == 'SMOTE':
      smp = SMOTE()
    elif sampling_type == 'ADASYN':
      smp = ADASYN()
    X_r, y_r = smp.fit_resample(X_train, y_train)
    return X_r, y_r

# ���ʂ̕\���i�����s��AROC�Ȑ���\���j
def disp_result(y_test, y_pred, sampling_type):
    target_names=['good', 'mal']
    cmx = confusion_matrix(y_test, y_pred, labels=[0,1])
    df_cmx = pd.DataFrame(cmx, index=target_names, columns=target_names)
    plt.figure(figsize = (2,2))
    # �����s����q�[�g�}�b�v�ŉ���
    hm = sns.heatmap(df_cmx,annot=True, cbar=False)
    plt.title('Confusion Matrix {}'.format(sampling_type))
    plt.savefig('conf_mat_{}.png'.format(sampling_type), dpi=500)
    plt.show()
    hm.get_figure().savefig('cmx_{}.png'.format(
                            sampling_type),
                            bbox_inches='tight')
    print(classification_report(y_test, y_pred,
   target_names=target_names))
    tn, fp, fn, tp = cmx.ravel()
    print( '{:<7}:\t{:>.3f}'.format('Accuracy', accuracy_score(y_test, y_pred)))
    print( '{:<7}:\t{:>.3f}'.format('Precision', tp / (tp + fp)))
    print( '{:<7}:\t{:>.3f}'.format('Recall', tp / (tp + fn)))
    # ROC�Ȑ��̂��߂�FPR, TPR���擾
    # �������lthreshold���擾 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # AUC(area under the curve)���v�Z
    auc_score = roc_auc_score(y_test, y_pred)
    # ROC�Ȑ���`��
    plt.figure(figsize=(4,3))
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc_score)
    plt.legend()
    plt.title('ROC Curve ({})'.format(sampling_type))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig('ROC_CURVE_{}.png'.format(sampling_type), 
  bbox_inches='tight', dpi=500)
    
def main():
    test_count = 100
    X_train, y_train, X_test, y_test, features = prepare(test_count)
    df_train = pd.DataFrame(X_train, columns=features)
    print(df_train)
    # ���T���v�����O���s�킸�ɁA�����_���t�H���X�g�Ŋw�K�E�\��
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    rf.fit(X_train, y_train)
    print('\nWithout Sampling')
    y_pred = rf.predict(X_test)
    disp_result(y_test, y_pred, 'without sampling')
    # 5��ނ̃��T���v�����O���s���A
    # �����_���t�H���X�g�Ŋw�K�E�\��
    for sampling_type in ['RUS', 'ENN', 
                           'ROS', 'SMOTE', 'ADASYN']:
        X_r, y_r = sampling(sampling_type, X_train, y_train)
        rf = RandomForestClassifier(max_depth=5, random_state=0)
        rf.fit(X_r, y_r)
        y_pred = rf.predict(X_test)
        disp_result(y_test, y_pred, sampling_type)

if __name__ == '__main__':
    main()
