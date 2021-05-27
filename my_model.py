from keras import Input, Model
from keras.layers import  Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,Concatenate, Dropout,Flatten, Lambda,Reshape
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
def slice_2(x, index):
    return x[:, index, 1:,:]
def slice(x, index):
    return x[:, index,0, :]
'''
class predict_model(object):#max_features为词典大小
    def __init__(self, embedding_dim=100, sym_embedding=128,dis_embedding=128,class_num=2,
                 last_activation='softmax'):
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.dis_embedding = dis_embedding
        self.sym_embedding = sym_embedding
        self.last_activation = last_activation
    def get_model(self):
        input = Input((2,6,self.embedding_dim,))#(?,2,6,100)
        conv=Conv1D(filters = 128,
                    kernel_size = 1,
                    activation = 'relu',
                    kernel_regularizer = l2(0.0),
                    kernel_initializer = 'glorot_uniform',
                    padding = 'valid',
                    strides = 1,
                    name = 'conv')
        conv2=Conv1D(filters = 128,
                    kernel_size = 2,
                    activation = 'relu',
                    kernel_regularizer = l2(0.0),
                    kernel_initializer = 'glorot_uniform',
                    padding = 'valid',
                    strides = 1,
                    name = 'conv2')
        self.dis_transform=Dense(self.dis_embedding, activation='relu',name='dis_transform')
        self.sym_transform=Dense(self.sym_embedding, activation='relu',name='sym_transform')#(?,5,128)
        all_dis=[]
        for i in range(0, 2):
            this_dis = Lambda(slice, output_shape=(100,), arguments={'index':i})(input)
            this_sym = Lambda(slice_2, output_shape=(5,100,), arguments={'index':i})(input)
            this_dis=self.dis_transform(this_dis)
            this_sym=self.sym_transform(this_sym)
            print("this_sym.shape",this_sym.shape)
            print("this_dis.shape",this_dis.shape)
            new_sym=conv(this_sym)#(?,1,128)
            print(new_sym.shape)
            new_sym=GlobalMaxPooling1D()(new_sym)#(?,128)
            print(new_sym.shape)
            print("new_sym.shape",new_sym.shape)
            this=[this_dis,new_sym]
            this = Concatenate()(this)
            print("this.shape",this.shape)#(?,256)
            all_dis.append(this)
        all_dis = Concatenate()(all_dis)
        all_dis = Reshape((2,self.dis_embedding+self.sym_embedding,))(all_dis)
        print("all_dis.shape",all_dis.shape)#(?,2,256)        c=conv(all_dis)#(?,1,128)
        c=conv2(all_dis)#(?,1,128)
        print(c.shape)
        c=GlobalMaxPooling1D()(c)#(?,128)
        print(c.shape)
        drop = Dropout(0.5)(c)
        print(drop.shape)
        output = Dense(self.class_num,activation=self.last_activation)(drop)
        print(output.shape)
        model = Model(inputs=input, outputs=output)
        return model
model = predict_model( ).get_model()
#需要将K.dot封装
'''
'''
class Self_Attention(Layer):
    def __init__(self, dis_embedding,sym_embedding,**kwargs):
        self.dis_embedding = dis_embedding
        self.sym_embedding = sym_embedding
        super(Self_Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, 2, 6,100)
        self.dis_transform=Dense(self.dis_embedding, activation='relu',name='dis_transform')
        self.sym_transform=Dense(self.sym_embedding, activation='relu',name='sym_transform')#(?,5,128)
        
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
 
    def call(self, x):
        all_dis=[]
        for i in range(0, 2):
            this_dis = Lambda(slice, output_shape=(100,), arguments={'index':i})(x)
            this_sym = Lambda(slice_2, output_shape=(5,100,), arguments={'index':i})(x)
            this_dis=self.dis_transform(this_dis)
            this_sym=self.sym_transform(this_sym)
            print("this_sym.shape",this_sym.shape)
            print("this_dis.shape",this_dis.shape)
            A = K.softmax(K.batch_dot(this_dis, K.permute_dimensions(this_sym, [0,2,1])))#(?,5)
            #A = Reshape((1,5,))(A)
            print("A.shape",A.shape)
            new_sym = K.batch_dot(A, this_sym)#(?,5)(?,5,128)--->(?,128)
            print("new_sym.shape",new_sym.shape)
            this=[this_dis,new_sym]
            this = Concatenate()(this)
            print("this.shape",this.shape)#(?,256)
            all_dis.append(this)
        all_dis = Concatenate()(all_dis)
        all_dis = Reshape((2,self.dis_embedding+self.sym_embedding,))(all_dis)
        print("all_dis.shape",all_dis.shape)#(?,2,256)
        return all_dis
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0],2,self.dis_embedding+self.sym_embedding)#(?,2,256)
class predict_model(object):#max_features为词典大小
    def __init__(self, embedding_dim=100, sym_embedding=128,dis_embedding=128,class_num=2,
                 last_activation='softmax'):
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.dis_embedding = dis_embedding
        self.sym_embedding = sym_embedding
        self.last_activation = last_activation
    def get_model(self):
        input = Input((2,6,self.embedding_dim,))#(?,2,6,100)
        conv=Conv1D(filters = 128,
                    kernel_size = 2,
                    activation = 'relu',
                    kernel_regularizer = l2(0.0),
                    kernel_initializer = 'glorot_uniform',
                    padding = 'valid',
                    strides = 1,
                    name = 'conv')
        all_dis=Self_Attention(self.dis_embedding,self.sym_embedding)(input)
        print("all_dis.shape",all_dis.shape)#(?,2,256)

        c=conv(all_dis)#(?,1,128)
        print(c.shape)
        c=GlobalMaxPooling1D()(c)#(?,128)
        print(c.shape)
        
        drop = Dropout(0.5)(c)
        print(drop.shape)
        output = Dense(self.class_num,activation=self.last_activation)(drop)
        print(output.shape)
        model = Model(inputs=input, outputs=output)
        return model
model = predict_model( ).get_model()
#model.summary()
'''
class predict_model(object):#max_features为词典大小
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















    
