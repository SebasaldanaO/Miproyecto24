np.random.seed(12345)
st.title('Calcula Tu Inmueble')

data = pd.DataFrame(datos,
                    columns = list('ABCDEF'))
st.dataframe(data)
e = np.random.normal(0,1,size=100)
y = data['price']*2 + data['assess']*3 + data['bdrms']*4 + data['lotsize']*5 + data['sqrft']*6 + data['colonial']*0.3 + 10 + e
model = DecisionTreeRegressor(max_depth = 4)
model.fit(data,y)
st.subheader('A')
val_a = st.slider('Seleccione El Valor De A',
          data['A'].min(),
          data['A'].max())
st.subheader('B')
val_b = st.slider('Seleccione El Valor De B',
          data['B'].min(),
          data['B'].max())
st.subheader('C')
val_c = st.slider('Seleccione El Valor De C',
          data['C'].min(),
          data['C'].max())
st.subheader('D')
val_d = st.slider('Seleccione El Valor De D',
          data['D'].min(),
          data['D'].max())

Valores = np.array([[val_price,val_assess,val_bdrms,val_lotsize,val_sqrft,val_colonial]])
pre = model.predict(Valores)
st.write(pre)

with open("model.picle", "rb") as f:
    model = pickle.load(f)

    