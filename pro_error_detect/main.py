import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Generate image caption function
def generate_caption(image, text=None):
    if text:
        inputs = processor(image, text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Image Captioning with BLIP", page_icon="🖼️")
    
    # Custom styles
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f7f7f7;
        }
        .stButton>button {
            background-color: #3b5998;
            color: white;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #2d4373;
        }
        .stTextInput>div>input {
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Image Captioning with BLIP Model 🖼️")
    st.write("Upload an image and enter optional text for conditional captioning.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        text = st.text_input("Optional text for conditional captioning:", "")
        
        if st.button("Generate Caption"):
            with st.spinner("Generating caption... please wait."):
                caption = generate_caption(image, text)
            st.write("Generated Caption:", caption)

if __name__ == "__main__":
    main()
