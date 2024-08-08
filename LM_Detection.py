import os
import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import random
from geopy.geocoders import Nominatim

# page_bg_img="""

# <style>
# [data-testid="stAppViewContainer"]{
#      background-image:url("https://unsplash.com/s/photos/taj-mahal")
#      background-size:cover;
# }
# [data-testid="stHeader"]{
#      background-color:rgba(0,0,0,0);

# }
# </style>
# """
# st.markdown(page_bg_img,unsafe_allow_html=True)


model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'

labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))



def image_processing(image):
    img_shape = (321, 321)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")])
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    result = classifier.predict(img)
    return labels[np.argmax(result)],img1

def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address,location.latitude, location.longitude









def run():
    




    st.title("Heritage Identification of Monuments")
    
    

    img = PIL.Image.open('logo.png')
    img = img.resize((256,256))
    st.image(img)
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
   
    save_dir = './Uploaded_Images/'
    os.makedirs(save_dir, exist_ok=True)

    if img_file is not None:
      save_image_path = './Uploaded_Images/' + img_file.name
      with open(save_image_path, "wb") as f:
          f.write(img_file.getbuffer())
      prediction, image = image_processing(save_image_path)

      
      st.image(image)
      st.header("📍 **Predicted Landmark is: " + prediction + '**')



    try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: '+address )

            # Add buttons for latitude, longitude, and map
            if st.button("Show Latitude"):
                st.write("Latitude:", latitude)

            if st.button("Show Longitude"):
                st.write("Longitude:", longitude)

            if st.button("Show Map"):
        # Open a new page or tab with the map
                st.markdown(f'<a href="http://www.google.com/maps?q={latitude},{longitude}" target="_blank">View Map</a>', unsafe_allow_html=True)

            # if st.button("Click to view map"):
            # # Open a new page or tab with the map
            #   st.markdown(f'<a href="http://www.google.com/maps?q={latitude},{longitude}" target="_blank">View Map</a>', unsafe_allow_html=True)


            # loc_dict = {'Latitude':latitude,'Longitude':longitude}
            # st.subheader('✅ **Latitude & Longitude of '+prediction+'**')
            # st.json(loc_dict)
            # data = [[latitude,longitude]]
            # df = pd.DataFrame(data, columns=['lat', 'lon'])
            # st.subheader('✅ **'+prediction +' on the Map**'+'🗺️')
            # st.map(df)
    except Exception as e:
            st.warning("Please Uplod The Image!!")
run()