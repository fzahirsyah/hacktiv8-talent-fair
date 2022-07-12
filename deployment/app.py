import streamlit as st
import pandas as pd
import pickle

st.title('Customer Clustering')
st.markdown('''
    These web Application is for predicting customer clustering using K-means Clustering
    from ecommerce dataset. there are 4 number of cluster that the model separates

    below are the example input for the form
''')
col = ['total_view', 'total_cart_add', 'total_purchases', 'total_sessions',
       'total_spent', 'min_spent', 'max_spent', 'cust_retailer_age',
       'days_since_last_activity', 'first_view_age', 'days_since_last_view',
       'first_cart_age', 'days_since_last_cart', 'first_purchase_age',
       'days_since_last_purchase']

st.text_input('costumer id',value=1195)
col1,col2 = st.columns(2)
total_view = col1.number_input('total view',min_value=0,max_value=1000,value=32,step=1)
total_cart_add = col2.number_input('total cart add',min_value=0,max_value=1000,value=1,step=1)
total_purchases = col1.number_input('total purchase',min_value=0,max_value=1000,value=3,step=1)
total_sessions = col2.number_input('total session',min_value=0,max_value=1000,value=18,step=1)
total_spent = col1.number_input('total spent',min_value=0,max_value=13000,value=310,step=1)
min_spent = col2.number_input('minimum spent',min_value=0,max_value=1000,value=20,step=1)
max_spent = col1.number_input('maximum spent',min_value=0,max_value=1000,value=95,step=1)
cust_retailer_age = col2.number_input('number of day from first costumer activity to last',min_value=0,max_value=1000,value=140,step=1)
days_since_last_activity = col1.number_input('days since last activity to current date',min_value=0,max_value=1000,value=130,step=1)
first_view_age = col2.number_input('number of day from first costumer view to last',min_value=0,max_value=1000,value=140,step=1)
days_since_last_view = col1.number_input('days since last view to current date',min_value=0,max_value=1000,value=140,step=1)
first_cart_age = col2.number_input('number of day from first add to cart to last',min_value=0,max_value=1000,value=140,step=1)
days_since_last_cart = col1.number_input('days since last add to cart to current date',min_value=0,max_value=1000,value=140,step=1)
first_purchase_age = col2.number_input('number of day from first purchase to last',min_value=0,max_value=1000,value=140,step=1)
days_since_last_purchase = col1.number_input('days since last purcase to current date',min_value=0,max_value=1000,value=140,step=1)


data_new = {
    'total_view':total_view,
    'total_cart_add':total_cart_add,
    'total_purchases':total_purchases,
    'total_sessions':total_sessions,
    'total_spent':total_spent,
    'min_spent':min_spent,
    'max_spent':max_spent,
    'cust_retailer_age':cust_retailer_age,
    'days_since_last_activity':days_since_last_activity,
    'first_view_age':first_view_age,
    'days_since_last_view':days_since_last_view,
    'first_cart_age':first_cart_age,
    'days_since_last_cart':days_since_last_cart,
    'first_purchase_age':first_purchase_age,
    'days_since_last_purchase':days_since_last_purchase
}
data_new = pd.DataFrame([data_new])

with open("model/model_kmean_cluster.pkl", "rb") as p:
    credit_score = pickle.load(p)



if st.button('predict'):
    predict_score = credit_score.predict(data_new)
    if predict_score == 1:
        st.subheader('Cluster 1 Best customer')
        st.markdown(" the best user averaging most spend and most purchase of the all cluster and averaging shortest time between last purchase")
    elif predict_score == 2:
        st.subheader('Cluster 2 Low value Costumer')
        st.markdown('the least user that spend in the ecommerce and averaging the longest days since last purchase')
    elif predict_score == 3:
        st.subheader('Cluster 3 Medium high Costumer')
        st.markdown('the 2nd tier of user spends in the ecommerce ')
    elif predict_score == 0:
        st.subheader('Cluster 0 Medium Low Costumer')
        st.markdown('''the 3rd tier of user spends in the ecommerce
        ''')