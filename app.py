import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from bs4 import BeautifulSoup
import json
from geopy.geocoders import Nominatim
import openmeteo_requests
import pandas as pd

# 1. INITIALIZATION
st.set_page_config(page_title="Tubonge Farm AI", layout="wide", page_icon="🌾")

class TubongeAI:
    def __init__(self):
        # Load disease model
        self.model = tf.lite.Interpreter('models/crop_disease_model.tflite')
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        
        # Initialize services
        self.geolocator = Nominatim(user_agent="tubonge_app")
        self.weather_client = openmeteo_requests.Client()
        
        # User session memory
        if 'history' not in st.session_state:
            st.session_state.history = []
            st.session_state.location = None

    # 2. CORE FUNCTIONALITY
    def greet_user(self):
        """Professional greeting with guidance"""
        return """
        Hello! I'm Tubonge Farm AI Assistant. I can:
        - Identify pests/diseases from plant images
        - Provide real-time farming advice
        - Show market prices and weather
        Please upload 3 leaf images (top, middle, bottom) + 1 whole plant image.
        """

    def diagnose_image(self, images):
        """Enhanced image analysis with guidance"""
        try:
            if len(images) < 4:
                return {
                    'status': 'incomplete',
                    'message': 'Please upload 4 images (3 leaves + 1 whole plant)'
                }
            
            # Process first leaf image
            img = Image.open(images[0]).resize((224, 224))
            img_array = np.expand_dims(np.array(img)/255.0, 0).astype(np.float32)
            
            self.model.set_tensor(self.input_details[0]['index'], img_array)
            self.model.invoke()
            pred = self.model.get_tensor(self.output_details[0]['index'])
            
            pest = self._get_class_name(np.argmax(pred[0]))
            return {
                'status': 'success',
                'pest': pest,
                'confidence': f"{np.max(pred[0]):.0%}",
                'solution': self._get_knowledge(pest),
                'prevention': self._get_prevention(pest)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Analysis failed: {str(e)}"
            }

    # 3. REAL-TIME KNOWLEDGE INTEGRATION
    def _get_knowledge(self, topic):
        """Dynamic knowledge from reliable sources"""
        sources = {
            "CABI": f"https://www.cabi.org/isc/?q={topic}",
            "Infonet": f"https://www.infonet-biovision.org/Search?search={topic}",
            "FAO": f"http://www.fao.org/faostat/en/#data/PP/visualize?query={topic}"
        }
        
        try:
            # Try CABI first
            response = requests.get(sources["CABI"], timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            result = soup.select('.result-title')[0].text if soup.select('.result-title') else ""
            if result: return f"{result[:250]}... (Source: CABI)"
            
            # Fallback to Infonet
            response = requests.get(sources["Infonet"], timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            result = soup.select('.search-results li')[0].text if soup.select('.search-results li') else ""
            return f"{result[:250]}... (Source: Infonet)" if result else self._get_fallback(topic)
        except:
            return self._get_fallback(topic)

    def _get_prevention(self, pest):
        """Prevention measures from curated knowledge"""
        prevention_db = {
            "fall_armyworm": "Rotate crops with legumes, use pheromone traps",
            "aphids": "Introduce ladybugs, spray neem oil weekly",
            # Add more pests
        }
        return prevention_db.get(pest, "Practice crop rotation and field sanitation")

    # 4. WEATHER & MARKET DATA
    def get_weather(self, location):
        try:
            geo = self.geolocator.geocode(location)
            params = {
                "latitude": geo.latitude,
                "longitude": geo.longitude,
                "current": ["temperature_2m", "precipitation"]
            }
            response = self.weather_client.weather_api(params)[0]
            return {
                "temperature": response.Current.Temperature_2m,
                "rain": response.Current.Precipitation,
                "location": location
            }
        except:
            return None

    def get_market_prices(self, crop, location="Nairobi"):
        try:
            # Mock API - replace with real source
            response = requests.get(f"https://api.fao.org/prices?crop={crop}&location={location}")
            prices = {
                "maize": {"Nairobi": 4500, "Kisumu": 4200},
                "tomatoes": {"Nairobi": 80, "Kisumu": 70}
            }
            return prices.get(crop, {}).get(location, "Data unavailable")
        except:
            return "Price service unavailable"

# STREAMLIT UI
def main():
    ai = TubongeAI()
    
    st.title("🌾 Tubonge Farm AI Assistant")
    st.markdown(ai.greet_user())
    
    # Location input
    location = st.text_input("Enter your location (for weather/market data):")
    if location:
        st.session_state.location = location
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Diagnose", "Knowledge", "Market"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload plant images (3 leaves + 1 whole plant):",
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            result = ai.diagnose_image(uploaded_files)
            
            if result['status'] == 'success':
                st.success(f"Detected: {result['pest']} ({result['confidence']} confidence)")
                st.write(f"**Solution:** {result['solution']}")
                st.write(f"**Prevention:** {result['prevention']}")
                
                # Show images
                cols = st.columns(4)
                for i, img in enumerate(uploaded_files[:4]):
                    cols[i].image(Image.open(img), caption=f"Image {i+1}")
            else:
                st.error(result['message'])
    
    with tab2:
        query = st.text_input("Ask any farming question:")
        if query:
            response = ai._get_knowledge(query)
            st.write(response)
    
    with tab3:
        if st.session_state.location:
            weather = ai.get_weather(st.session_state.location)
            if weather:
                st.metric("Current Temperature", f"{weather['temperature']}°C")
                st.metric("Precipitation", f"{weather['rain']}mm")
            
            crop = st.selectbox("Select crop:", ["maize", "tomatoes", "beans"])
            price = ai.get_market_prices(crop, st.session_state.location)
            st.metric(f"Current {crop} price", f"KSh {price}")

if __name__ == "__main__":
    main()
