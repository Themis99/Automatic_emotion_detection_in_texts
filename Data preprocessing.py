import pandas as pd

def minmax_len(data):

    len_ = []
    for text in data['text']:
        len_.append(len(text))

    max_text = max(len_)
    min_text = min(len_)

    return max_text,min_text

if __name__ == '__main__':
    #prepare train set
    train_data = pd.read_csv('C:\\Users\\Themis\\Desktop\\Github projects\\Emotion classifier\\Dataset\\train.txt',sep=';')
    train_data.columns = ['text','class']

    #prepare test set
    test_data = pd.read_csv('C:\\Users\\Themis\\Desktop\\Github projects\\Emotion classifier\\Dataset\\test.txt',sep=';')
    test_data.columns = ['text','class']

    #prepare val set
    val_set = pd.read_csv('C:\\Users\\Themis\\Desktop\\Github projects\\Emotion classifier\\Dataset\\val.txt',sep=';')
    val_set.columns = ['text','class']

#size of every dataset
    print('size for the train set: \n',train_data.shape)
    print('size for the test set: \n',test_data.shape)
    print('size for the val set: \n',val_set.shape)

    #distribution of classes
    print('distribution of classes\n')

    train_count = train_data['class'].value_counts()
    print('distribution for train set: \n',train_count)

    test_count = test_data['class'].value_counts()
    print('distribution for test set: \n',test_count)

    val_count = test_data['class'].value_counts()
    print('distribution for val set: \n',val_count)

    # min max lenght for all texts in all datasets
    max_,min_ = minmax_len(train_data)
    print('max train: \n',max_)
    print('min train: \n',min_)

    max_,min_ = minmax_len(test_data)
    print('max test: \n',max_)
    print('min test: \n',min_)

    max_,min_ = minmax_len(val_set)
    print('max val: \n',max_)
    print('min val: \n',min_)









