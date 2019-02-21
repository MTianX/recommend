import keras
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
# print("训练集样本长度为：{},测试集样本长度为{}".format(len(x_train),len(x_test)))
def one_hot(list,dim):
    result = np.zeros(shape=(dim),dtype=np.int32)
    for i in list:
        result[i] = 1
    return result


def create_onehot(list,dim):
    result = np.zeros(shape=[len(list),dim],dtype=np.int32)
    for i,val in enumerate(list):
        for index in val:
            result[i][index] = 1
    return result

def Create_data():
    ratings_path = r'D:\study\T\MachineLearn\keras\recommend\ml-1m\ratings.dat'
    ratings_col_name = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings_pd = pd.read_csv(ratings_path,names=ratings_col_name,sep='::',engine='python')
    ratings_pd = ratings_pd.filter(regex='UserID|MovieID|Rating')

    user_path = r'D:\study\T\MachineLearn\keras\recommend\ml-1m\users.dat'
    user_col_name =  ['UserID','Gender','Age','Job','Zip-code']
    user_pd = pd.read_csv(user_path,names=user_col_name,sep="::",engine='python')
    user_pd = user_pd.filter(regex='UserID|Gender|Age|Job')
    gender_dict = {'F':0,'M':1}
    gender_map = { val:gender_dict[val] for i,val in enumerate(user_pd['Gender'])}
    user_pd['Gender'] = user_pd['Gender'].map(gender_map)
    age_dict = {1:0,18:1,25:2,35:3,45:4,50:5,56:6}
    age_map = { val:age_dict[val] for i,val in enumerate(user_pd['Age'])}
    user_pd['Age'] = user_pd['Age'].map(age_map)
    newdata = pd.merge(ratings_pd,user_pd)

    movie_path = r'D:\study\T\MachineLearn\keras\recommend\ml-1m\movies.dat'
    movie_col_name = ['MovieID','Title','Genres']
    movie_pd = pd.read_csv(movie_path,names=movie_col_name,sep='::',engine='python')
    pattern = re.compile(r'\s[(][0-9]\d*[)]$')
    title_map = {val: re.sub(pattern, repl='', string=val) for i, val in enumerate(movie_pd['Title'])}
    movie_pd['Title'] = movie_pd['Title'].map(title_map)

    Genres_dic = {
        'Action': 0, 'Adventure': 1, 'Animation': 2, "Children's": 3, 'Comedy': 4, 'Crime': 5, 'Documentary': 6,
        'Drama': 7,
        'Fantasy': 8,
        'Film-Noir': 9, 'Horror': 10, 'Musical': 11, 'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 'Thriller': 15,
        'War': 16,
        'Western': 17
    }
    GenresMap = {val: [Genres_dic[ii] for ii in val.split('|')] for i, val in enumerate(movie_pd['Genres'])}
    movie_pd['Genres'] = movie_pd['Genres'].map(GenresMap)

    newdata = pd.merge(newdata,movie_pd)

    features_pd,targets_pd = newdata.drop('Rating',axis=1),newdata['Rating']
    # features_pd 列名UserID|MovieID|Gender|Age|Job|Title|Genres
    # targets_pd 列名 Rating
    features = features_pd.values
    targets = targets_pd.values
    x_train, x_test, y_train, y_test = train_test_split(features,targets,test_size=0.2,random_state=10)
    print("训练集样本长度为：{},测试集样本长度为{}".format(len(x_train),len(x_test)))
    return x_train, x_test, y_train, y_test

def net(x_train, x_test, y_train, y_test):

    train_user_id = np.array(x_train[:, 0]).astype(dtype=np.int32)
    train_user_age = np.array(x_train[:, 3]).astype(dtype=np.int32)
    train_user_job = np.array(x_train[:, 4]).astype(dtype=np.int32)
    train_user_gender = np.array(x_train[:, 2]).astype(dtype=np.int32)
    train_movie_id = np.array(x_train[:, 1]).astype(dtype=np.int32)
    train_movie_genres = create_onehot(x_train[:, 6],dim=18)

    test_user_id = np.array(x_test[:,0]).astype(dtype=np.int32)
    test_user_age = np.array(x_test[:, 3]).astype(dtype=np.int32)
    test_user_job = np.array(x_test[:, 4]).astype(dtype=np.int32)
    test_user_gender = np.array(x_test[:, 2]).astype(dtype=np.int32)
    test_movie_id = np.array(x_test[:, 1]).astype(dtype=np.int32)
    test_movie_genres = create_onehot(x_test[:, 6],dim=18)
    print(train_user_age)


    #用户数据长度
    uers_num = 6041
    user_age_num = 7
    user_job_num = 21
    user_gender_num = 2
    user_output_dim = 32

    #创建用户输入
    user_id = keras.layers.Input(shape=(1,),dtype=np.int32,name='user_id')
    user_age = keras.layers.Input(shape=(1,),dtype=np.int32,name='user_age')
    user_job = keras.layers.Input(shape=(1,),dtype=np.int32,name='user_job')
    user_gender = keras.layers.Input(shape=(1,),dtype=np.int32,name='user_gender')

    #创建用户嵌入层
    user_id_out = keras.layers.Embedding(input_dim=uers_num,output_dim=user_output_dim,input_length=1)(user_id)
    user_id_out = keras.layers.Reshape((user_output_dim,))(user_id_out)
    user_age_out = keras.layers.Embedding(input_dim=user_age_num,output_dim=user_output_dim,input_length=1)(user_age)
    user_age_out = keras.layers.Reshape((user_output_dim,))(user_age_out)
    user_job_out = keras.layers.Embedding(input_dim=user_job_num,output_dim=user_output_dim,input_length=1)(user_job)
    user_job_out = keras.layers.Reshape((user_output_dim,))(user_job_out)
    user_gender_out = keras.layers.Embedding(input_dim=user_gender_num,output_dim=user_output_dim,input_length=1)(user_gender)
    user_gender_out = keras.layers.Reshape((user_output_dim,))(user_gender_out)

    user_input = keras.layers.Concatenate()([user_id_out,user_age_out,user_job_out,user_gender_out])

    user_model = keras.layers.Dense(128)(user_input)
    user_model = keras.layers.Dense(256)(user_model)
    user_model = keras.layers.Reshape((256,))(user_model)

    #创建电影输入
    movie_id_num = 3953
    movie_genres_num = 18
    movie_output_dim = 32

    movie_id = keras.layers.Input(shape=(1,),dtype=np.int32,name='movie_id')
    movie_genres = keras.layers.Input(shape=(18,),name='movie_genres')

    movie_id_out = keras.layers.Embedding(input_dim=movie_id_num,output_dim=movie_output_dim,input_length=1)(movie_id)
    print(movie_id_out)
    movie_id_out = keras.layers.Reshape((movie_output_dim,))(movie_id_out)
    print(movie_id_out)

    movie_genres_out = keras.layers.Embedding(input_dim=movie_genres_num,output_dim=movie_output_dim,input_length=18)(movie_genres)
    print(movie_genres_out)
    movie_genres_out = keras.layers.Flatten()(movie_genres_out)
    print(movie_genres_out)
    movie_genres_out = keras.layers.Dense(movie_output_dim)(movie_genres_out)
    print(movie_genres_out)
    # movie_genres_out = keras.layers.Reshape((movie_output_dim,))(movie_genres_out)
    # print(movie_genres_out)
    movie_input = keras.layers.Concatenate()([movie_id_out,movie_genres_out])

    movie_model = keras.layers.Dense(128)(movie_input)
    movie_model = keras.layers.Dense(256)(movie_model)
    movie_model = keras.layers.Reshape((256,))(movie_model)

    out = keras.layers.Multiply()([user_model,movie_model])
    out = keras.layers.Dense(256,activation='relu')(out)
    out = keras.layers.Dropout(0.5)(out)
    out = keras.layers.Dense(1,activation='linear')(out)

    model = keras.Model([user_id,user_age,user_job,user_gender,movie_id,movie_genres],out)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
    # keras.utils.plot_model(model,to_file='model.png')
    # print(model.summary())

    # model.fit(x=[train_user_id,train_user_age,train_user_job,train_user_gender,train_movie_id,train_movie_genres],
    #           y=y_train,batch_size=100,epochs=12)

    model.fit(x=[train_user_id,train_user_age,train_user_job,train_user_gender,train_movie_id,train_movie_genres],
              y=y_train,batch_size=1000,epochs=6,validation_data=[[test_user_id,test_user_age,test_user_job,test_user_gender,test_movie_id,test_movie_genres],
                                                                    y_test],shuffle=False)
    model.save('model.h5')


def perd(selectdata):

    user_id = np.array(selectdata[0])
    user_age = np.array(selectdata[3])
    user_job = np.array(selectdata[4])
    user_gender = np.array(selectdata[2])
    movie_id = np.array(selectdata[1])
    movie_genres = one_hot(selectdata[6], dim=18)

    model = keras.models.load_model('model.h5')
    print(list)
    y = model.predict([user_id,user_age,user_job,user_gender,movie_id,movie_genres])
    print('预测值:{}'.format(y))

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = Create_data()
    # index = 1
    # selectdata = x_test[index,:]
    # y = y_test[index]
    #
    #
    #
    # print("真实值:{}".format(y))
    net(x_train, x_test, y_train, y_test)
    # perd(selectdata)

