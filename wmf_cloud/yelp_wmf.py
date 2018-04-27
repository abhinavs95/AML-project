import pandas as pd
import numpy as np
import scipy.sparse
import wmf
import scipy.sparse.linalg as sla
import math

def itemize(data):
    _,indx = np.unique(data,return_index=True)
    u = data[np.sort(indx)]
    n_data = u.shape[0]
    new_indx = np.arange(n_data)
    d = dict(zip(u,new_indx))
    data_indx = np.zeros(data.shape, dtype=np.int32)
    for i in range(data_indx.shape[0]):
        data_indx[i] = d[data[i]]
    return data_indx, n_data

def load_data():
    data = pd.read_csv('review.csv')
    data = data.drop(['funny', 'review_id', 'text', 'date', 'useful', 'cool'], axis=1)
    data.drop_duplicates(inplace=True)
    rows, cols, stars = np.array(data['user_id']), np.array(data['business_id']), np.array(data['stars'],dtype=np.uint8)
    cols = cols[stars>=3]
    rows = rows[stars>=3]
    stars = stars[stars>=3]
    # itemize users and items
    row_indx, n_users = itemize(rows)
    col_indx, n_items = itemize(cols)
    return scipy.sparse.csr_matrix((stars,(row_indx, col_indx)), dtype=np.uint8, shape=(n_users,n_items))


def rmse(R,V,W):
    num = R.count_nonzero()
    term = scipy.sparse.lil_matrix(R.shape)
    ind = R.nonzero()
    i_prev=0
    k=0
    for i in range(len(ind[0])-1):
        if ind[0][i]==ind[0][i+1]:
            i=i+1
        else:
            term[k,ind[1][i_prev:i+1]] = np.asarray([np.dot(V[ind[0][i],:],W[j,:]) for j in ind[1][i_prev:i+1]])
            i_prev=i+1
            k=k+1
    term[k,ind[1][i_prev:i+1]] = np.asarray([np.dot(V[ind[0][i],:],W[j,:]) for j in ind[1][i_prev:i+1]])
    term = R - term
    rmse = math.sqrt(sla.norm(term)**2/num)
    return rmse

def rmse_new(R,U,V):
    num = R.count_nonzero()
    term = R - np.dot(U,V.T)
    rmse = math.sqrt(sla.norm(term)**2/num)
    return rmse

# Function for loading and pre-processing data
def get_data(path):
    np_frame = pd.read_csv(path,usecols=[0,1,2]).as_matrix()

    users = np_frame[:,0].astype(dtype = 'uint32')-1
    items = np_frame[:,1].astype(dtype = 'uint32')-1
    ratings = np_frame[:,2].astype(dtype = 'float64')

    items_unique, items_cleaned = np.unique(items,return_inverse=True)

    num_users = np.max(users)+1
    num_items = items_unique.shape[0]

    user_dict = {i: [] for i in range(num_users)}

    for i in range(len(users)):
        user_dict[users[i]].append([items_cleaned[i],ratings[i]])

    R_train = scipy.sparse.lil_matrix((num_users,num_items))
    R_test = scipy.sparse.lil_matrix((num_users,num_items))
    num_train = 0
    num_test = 0

    for i in user_dict.keys():
        l = len(user_dict[i])
        indx = np.arange(l)
        np.random.shuffle(indx)
        temp = np.asarray(user_dict[i])
        R_test[i,temp[indx[:l/2],0]] = temp[indx[:l/2],1]
        R_train[i,temp[indx[l/2:],0]] = temp[indx[l/2:],1]
        num_train += len(indx[l/2:])
        num_test += len(indx[:l/2])

    return R_train.tocsr()

#path = 'ratings.csv'
#R = get_data(path)

R = load_data()
R = R[:-1000000,:-100000]
R.data = np.ones_like(R.data)
S = wmf.log_surplus_confidence_matrix(R, alpha=20.0, epsilon=1e-6)

num_iters = 10
num_factors = 50

U,V = wmf.factorize(S,num_factors,R=R,num_iterations=num_iters, verbose=True)

print('rmse',rmse(R,U,V))
np.save('U_wmf_2',U)
np.save('V_wmf_2',V)

