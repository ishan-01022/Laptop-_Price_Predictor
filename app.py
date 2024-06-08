import streamlit as st
import numpy as np
import pickle

pipe = pickle.load(open('model.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.write("""
    ## Welcome to Laptop Price Prediction Tool
    Unveil ***laptop pricing magic***! Share specs & predict your tech companion's cost.
""")

company = st.selectbox('Brand', ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
                                 'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
                                 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'])

typename = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible',
                                 'Workstation'])

ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Weight', min_value=0.5, max_value=10.0, value=1.0)

ts = st.selectbox('Touchscreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size', min_value=10, max_value=20, value=15)

resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440'])

cpu = st.selectbox('CPU', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3',
                           'Other Intel Processor'])

hdd = st.selectbox('HDD(in GB)', [128, 256, 512, 1024, 2048, 0])

ssd = st.selectbox('SSD(in GB)', [8, 128, 256, 512, 1024, 0])

gpu = st.selectbox('GPU', ['Intel', 'AMD', 'Nvidia'])

os = st.selectbox('OS', ['Mac', 'Others / No Os / Linux', 'Windows'])

if st.button('Predict Price'):
    ts = 1 if ts == "Yes" else 0

    ips = 1 if ts == "Yes" else 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, typename, ram, weight, ts, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title("Predicted Value for this configuration is:" + str(int(np.exp(pipe.predict(query)[0]))))

