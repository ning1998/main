from keras import Input, Model
from keras.layers import  Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,Concatenate, Dropout,Flatten, Lambda,Reshape
from keras.regularizers import l2
from keras import backend as K

def slice_2(x, index):
    return x[:, index, :,:]
def slice(x, index):
    return x[:, index, :]

class TextCNN(object):#max_features为词典大小
    def __init__(self, embedding_dim=100, dis_embedding=128,class_num=2,
                 last_activation='softmax'):
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.dis_embedding = dis_embedding
        self.last_activation = last_activation
    def get_model(self):
        input = Input((2,self.embedding_dim,))#(?,2,100)
        Dense1=Dense(self.dis_embedding, activation='relu',name='Dense1')#对输入向量进行转换，投射到并发关系的空间
        conv=Conv1D(filters = 128,
                    kernel_size = 2,
                    activation = 'relu',
                    kernel_regularizer = l2(0.0),
                    kernel_initializer = 'glorot_uniform',
                    padding = 'valid',
                    strides = 1,
                    name = 'conv')
       
        all_dis=[]
        for i in range(0, 2):
            this_embed = Lambda(slice, output_shape=(self.embedding_dim,), arguments={'index':i})(input)
            #print(this_embed.shape)#(?,100)
            this_dis=Dense1(this_embed)
            all_dis.append(this_dis)
        all_dis = Concatenate()(all_dis)
        all_dis = Reshape((2,self.dis_embedding,))(all_dis)
        print(all_dis.shape)

        c=conv(all_dis)#(?,1,128)
        c=GlobalMaxPooling1D()(c)#(?,128)
        
        drop = Dropout(0.5)(c)
        output = Dense(self.class_num,activation=self.last_activation)(drop)
        model = Model(inputs=input, outputs=output)
        return model

#model = TextCNN( ).get_model()
#model.summary()




















    
