import streamlit as st
import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import base64
from PIL import Image
from streamlit_option_menu import option_menu
import io  # Import for creating in-memory file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tempfile  # Import for creating temporary files
import time

def set_bg_style():
    st.markdown(
        """
        <style>
            body, .stApp {
                background-color: white !important;
                color: black !important;
            }
            .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stFileUploader, .stImage, .stAlert, .stRadio, .stColumn, .stTextArea, .stSlider, .stNumberInput, .stText, .stTitle, .stHeader, .stCaption, .stExpander, .stCodeBlock, .stDataFrame, .stTable {
                color: black !important;
            }
            div[data-testid="stMarkdownContainer"] {
                color: black !important;
            }
            h1, h2, h3, h4, h5, h6, p, label, span {
                color: black !important;
            }
            .css-1aumxhk {
                color: black !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# Define a function for the home page
def show_home():
    st.markdown('<h1 style="text-align: center;">DETECT WELL</h1>', unsafe_allow_html=True)

    st.write("""
    This web application assists in detecting brain tumors, brest cancer, kidney cancer, lungs cancer from MRI scans using machine learning.

🔹 Multi-Cancer Detection – Detects different types of tumors, including brain, breast, kidney, and lung cancer.

🔹 MRI Image Upload – Users can upload MRI scans for automated analysis.

🔹 AI-Powered Predictions – Utilizes advanced machine learning models for accurate tumor classification.

🔹 Fast and Reliable – Provides instant predictions with high precision.

🔹 Easy-to-Use Interface – Simple and intuitive design for seamless user experience.

🔹 Detailed Insights – Displays results with explanations of different cancer types.

🔹 Medical Guidance – Offers basic information about symptoms and diagnosis.

🔹 Secure and Private – Ensures confidentiality of uploaded images.

   
    """)
    col1, col2 = st.columns((1, 1))
    with col1:
        image_path =r"C:\Users\HP\Desktop\1_aSC3odScNMyz7Y6MZvqJ1Q.jpg"

        # Open the image using PIL
        img = Image.open(image_path)

        # Resize the image
        img = img.resize((600, 500))

        # Display the image
        st.image(img)
        st.info("Brain tomor")
        st.write("A brain tumor is an abnormal mass of cells in the brain, which can be benign (non-cancerous) or malignant (cancerous). Symptoms vary based on size and location, including headaches, seizures, memory loss, vision issues, and speech difficulties. Causes may include genetic mutations, radiation exposure, or environmental factors**. Diagnosis involves MRI, CT scans, and biopsies. Treatment depends on the tumor type and may include surgery, radiation therapy, chemotherapy, or targeted drug therapy. Early detection improves outcomes, making regular medical checkups crucial for those at risk.")
        
        image_path =r"C:\Users\HP\Desktop\images (3).jpg"

        # Open the image using PIL
        img = Image.open(image_path)

        # Resize the image
        img = img.resize((600, 500))

        # Display the image
        st.image(img)
        st.info("Kidney cancer")
        st.write("Kidney cancer, also known as renal cancer, occurs when abnormal cells grow uncontrollably in the kidneys. The most common type is renal cell carcinoma (RCC). Symptoms may include blood in the urine, lower back pain, unexplained weight loss, and fatigue. Risk factors include smoking, obesity, high blood pressure, and genetic conditions. Diagnosis involves imaging tests like CT scans and MRIs, along with biopsies. Treatment options include surgery, targeted therapy, immunotherapy, and radiation. Early detection improves survival rates, making regular checkups essential. A healthy lifestyle, including a balanced diet and exercise, may help lower the risk.")


    with col2:
        image_path =r"C:\Users\HP\Desktop\images (2).jpg"
        # Open the image using PIL
        img = Image.open(image_path)

        # Resize the image
        img = img.resize((600, 500))

        # Display the image
        st.image(img)
        st.info("Breast cancer")
        st.write("Breast cancer is a disease in which abnormal cells in the breast grow uncontrollably. It can occur in both men and women, though it is more common in women. Early detection through mammograms and self-exams improves treatment outcomes. Symptoms include lumps, skin changes, or nipple discharge. Risk factors include genetics, age, and lifestyle. Treatment options vary from surgery, chemotherapy, and radiation to targeted therapies. Regular screenings and a healthy lifestyle can help in prevention. Awareness and research continue to improve survival rates and treatment advancements. Always consult a doctor for any concerns.")
        image_path =r"C:\Users\HP\Desktop\istockphoto-530196490-612x612.jpg"
        # Open the image using PIL
        img = Image.open(image_path)

        # Resize the image
        img = img.resize((600, 500))

        # Display the image
        st.image(img)
        st.info("Lung cancer")
        st.write("Lung cancer is a disease where abnormal cells grow uncontrollably in the lungs, often due to smoking, pollution, or genetic factors. It is one of the leading causes of cancer-related deaths worldwide. Symptoms include persistent cough, chest pain, shortness of breath, and weight loss. Early detection through imaging and biopsies improves survival rates. Treatment options include surgery, chemotherapy, radiation therapy, and targeted drugs. Quitting smoking and reducing exposure to harmful pollutants can lower the risk. Advances in immunotherapy are offering new hope for patients, improving outcomes and quality of life. Regular screenings help in early diagnosis.")

# Define a function for the brain tumor detection page
import streamlit as st
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import base64
# PDF report imports
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
import tempfile
from PIL import Image  # Import PIL here



# Define a function for the brain tumor detection page
def show_braintumor():
    
    st.markdown(
    """
    <style>
        .stAlert {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
    )


    st.markdown("<h1 style='text-align: center;'>Brain Tumor Detection</h1>", unsafe_allow_html=True)
    st.warning(" Warning: Kindly ensure that only brain tumor-related messages/files are uploaded. Thank you!", icon="⚠")
    st.warning(" Warning: after every use if you want to use again please reload web app. Thank you!", icon="⚠")

    classes = {"pituitary_tumor_br": 0,"no_tumor_br": 1,"meningioma_tumor_br": 2,"glioma_tumor_br": 3}
    dec = {0: 'Pituitary tumor detected', 1: "No tumor detected",2: "Meningioma tumor detected",3: "Glioma tumor detected" }
    button_style = f"""
        <style>
        div.stButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        div.stDownloadButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stDownloadButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
                 font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        </style>
        """
    st.markdown(button_style, unsafe_allow_html=True)


    def load_data(data_dir):
        x, y = [], []
        for cls in classes:
            pth = os.path.join(data_dir, cls)
            if os.listdir(pth):
                for img_name in os.listdir(pth):
                    try:
                        img = cv2.imread(os.path.join(pth, img_name), cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                            img = cv2.resize(img, (100, 100))
                            x.append(img)
                            y.append(classes[cls])
                    except Exception as e:
                        st.error(f"Error loading image {img_name}: {e}")
        return np.array(x), np.array(y)


    data_dir = "C:/Users/hp/Documents/braintumordetection/braintumordetection"
    x, y = load_data(data_dir)


    x_flat = x.reshape(len(x), -1) / 255.0


    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42)


    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)


    lg = LogisticRegression(C=0.1)
    lg.fit(x_train_pca, y_train)
   
    st.write("Welcome to our MRI Detection web application! This innovative tool utilizes advanced machine learning algorithms to analyze MRI scans and detect the presence of tumors. Developed by a team of medical imaging experts and data scientists, our app offers a reliable and efficient solution for early diagnosis and treatment planning.")

    col1, col2 = st.columns((1, 1))
    prediction_made = False
    symptoms_text = None
    diagnosis_text = None
    treatment_text = None
    additional_text = None
    uploaded_image = None
    prediction = None

    with col1:
        
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:

            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            uploaded_image = img # Save uploaded image
            if img is not None:
                img_resized = cv2.resize(img, (100, 100))
                img_normalized = img_resized / 255.0
                img_input = img_normalized.reshape(1, -1)
                img_pca = pca.transform(img_input)
                prediction_lg = lg.predict(img_pca)


                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode()
                st.markdown(f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{img_base64}" alt="Uploaded Image" width="200"></div>',
                            unsafe_allow_html=True
                           )
                prediction = dec[prediction_lg[0]]
                st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)

    with col2:
        if prediction is not None:

            if prediction == 'Pituitary tumor detected':
                st.info("Symptoms")
                symptoms_text = "Symptoms of pituitary tumors vary based on the type and size of the tumor, as well as the hormones affected. Common symptoms include: Headaches, Vision problems, especially if the tumor presses on the optic nerves, Nausea and vomiting, Hormonal imbalances that may lead to conditions like: Unexplained lactation in women, Reduced libido or fertility issues in men"
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "Diagnosing a pituitary tumor typically involves: A thorough medical history and physical examination. Imaging tests such as MRI or CT scans to visualize the tumor. Blood tests to assess hormone levels, which can indicate whether the pituitary gland is functioning normally or producing excess hormones."
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Treatment for a pituitary brain tumor depends on its size and effects. Doctors may use medicines to shrink certain tumors, surgery to remove them safely, or radiation to stop their growth. If the tumor affects hormone levels, patients might need hormone therapy. Small, harmless tumors may just be watched over time."
                st.write(treatment_text)
                
            elif prediction == 'Meningioma tumor detected':  # Corrected here
                st.info("Symptoms")
                symptoms_text = "Many meningiomas are asymptomatic until they reach a significant size. When symptoms do occur, they may include: Headaches, Seizures, Vision problems, Neurological deficits such as weakness or numbness, Hormonal changes if located near endocrine structures like the pituitary gland."
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "Diagnosis typically involves imaging techniques such as MRI or CT scans to visualize the tumor's size and location. In some cases, a biopsy may be performed to determine the tumor's grade and type."
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Meningioma tumors grow in the brain’s protective layers and are usually slow-growing and non-cancerous. Treatment depends on the size and symptoms. Small, harmless tumors may just need regular check-ups. If the tumor causes problems like headaches or vision issues,surgery is often done to remove it. If surgery isn’t possible, radiation therapy can help shrink or stop its growth. In rare cases, medication may be used if the tumor is aggressive."
                st.write(treatment_text)

            elif prediction == 'Glioma tumor detected':  # Corrected here
                st.info("Symptoms")
                symptoms_text = "Symptoms of gliomas depend on their size and location but may include: Headaches, Seizures, Cognitive changes or memory problems, Motor function impairment, Vision or hearing issues."
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "Diagnosis typically involves imaging techniques such as MRI or CT scans to visualize the tumor's size and location. A biopsy may be performed to determine the tumor's type and grade through histological examination."
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Glioma tumor treatment depends on its size, location, and growth rate. Doctors may use surgery to remove as much of the tumor as possible, followed by radiation therapy or chemotherapy to kill remaining cancer cells. In some cases, targeted drug treatments or therapy to manage symptoms like headaches or seizures may also be used. Regular check-ups help monitor progress and adjust treatment as needed."
                st.write(treatment_text)
                
            elif prediction == 'No tumor detected':
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)

            else:
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)
    if prediction is not None:
        st.markdown("<h3 style='text-align: center;'>Thank you for using our application!</h3>", unsafe_allow_html=True)

    # Report Generation and Download
        if st.button("Generate and Download Report"):
            # Create a PDF in memory
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Report Title
            c.setFont('Helvetica-Bold', 16)
            c.drawString(100, 750, "Brain Tumor Detection Report")

            # Uploaded Image (if available)
            if uploaded_image is not None:
                try:
                    from PIL import Image  # Import PIL here

                    # Convert the OpenCV image to PIL format
                    pil_img = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
                    pil_img = Image.fromarray(pil_img)
                    pil_img = pil_img.convert("RGB")  # Ensure RGB color mode

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_img.save(tmp_file, "PNG")
                        temp_image_path = tmp_file.name

                    # Use the temporary file path with drawImage
                    c.drawImage(temp_image_path, 200, 590, width=200, height=150, mask='auto')
                    c.drawString(100, 570, "Uploaded Image:")

                except Exception as e:
                    c.setFont('Helvetica-Oblique', 10)
                    c.drawString(100, 580, f"Error embedding image: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            # Prediction Result
            c.setFont('Helvetica-Bold', 12)
            c.drawString(100, 550, f"Prediction: {prediction}")

            # Symptoms and Diagnosis
            style = styles["Normal"]
            style.leading = 14  # Adjust line spacing

            y_position = 530
            left_margin = 100
            text_width = 400  # Adjust as needed

            def add_paragraph(text, y_pos):
                p = Paragraph(text, style)
                p.wrapOn(c, text_width, 1000)
                p.drawOn(c, left_margin, y_pos - p.height)
                return y_pos - p.height - 10  # Update y_position


            if symptoms_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Symptoms:")
                y_position -= 15
                y_position = add_paragraph(symptoms_text, y_position)

            if diagnosis_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Diagnosis:")
                y_position -= 15
                y_position = add_paragraph(diagnosis_text, y_position)

            if additional_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Additional Information:")
                y_position -= 15
                y_position = add_paragraph(additional_text, y_position)
                
            if treatment_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Treatment:")
                y_position -= 15
                y_position = add_paragraph(treatment_text, y_position)



            c.save()
            buffer.seek(0)

            st.download_button(
                label="Download Report",
                data=buffer,
                file_name="brain_tumor_report.pdf",
                mime="application/pdf",
            )
def show_brestcancer():
    
    st.markdown(
    """
    <style>
        .stAlert {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center;'>Breast Cancer Detection</h1>", unsafe_allow_html=True)
    st.warning(" Warning: Kindly ensure that only breast cancer-related messages/files are uploaded. Thank you!", icon="⚠")
    st.warning(" Warning: after every use if you want to use again please reload web app. Thank you!", icon="⚠")
    classes = {"benign_b": 0,"normal_b": 1,"malignant_b":2}
    dec = {0: 'Benign breast cancer detected', 1: "No tumor detected",2:"Malignant" }
    button_style = f"""
        <style>
        div.stButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        div.stDownloadButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stDownloadButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
                 font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        </style>
        """
    st.markdown(button_style, unsafe_allow_html=True)


    def load_data(data_dir):
        x, y = [], []
        for cls in classes:
            pth = os.path.join(data_dir, cls)
            if os.listdir(pth):
                for img_name in os.listdir(pth):
                    try:
                        img = cv2.imread(os.path.join(pth, img_name), cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                            img = cv2.resize(img, (100, 100))
                            x.append(img)
                            y.append(classes[cls])
                    except Exception as e:
                        st.error(f"Error loading image {img_name}: {e}")
        return np.array(x), np.array(y)


    data_dir = "C:/Users/hp/Documents/braintumordetection/braintumordetection"#changed the directory
    x, y = load_data(data_dir)


    x_flat = x.reshape(len(x), -1) / 255.0


    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42)


    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)


    lg = LogisticRegression(C=0.1)
    lg.fit(x_train_pca, y_train)
   
    st.write("Welcome to our MRI Detection web application! This innovative tool utilizes advanced machine learning algorithms to analyze MRI scans and detect the presence of tumors. Developed by a team of medical imaging experts and data scientists, our app offers a reliable and efficient solution for early diagnosis and treatment planning.")

    col1, col2 = st.columns((1, 1))
    prediction_made = False
    symptoms_text = None
    diagnosis_text = None
    additional_text = None
    treatment_text = None
    uploaded_image = None
    prediction = None

    with col1:
        
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:

            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            uploaded_image = img # Save uploaded image
            if img is not None:
                img_resized = cv2.resize(img, (100, 100))
                img_normalized = img_resized / 255.0
                img_input = img_normalized.reshape(1, -1)
                img_pca = pca.transform(img_input)
                prediction_lg = lg.predict(img_pca)


                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode()
                st.markdown(f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{img_base64}" alt="Uploaded Image" width="200"></div>',
                            unsafe_allow_html=True
                           )
                prediction = dec[prediction_lg[0]]
                st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)

    with col2:
        if prediction is not None:

            if prediction == 'Benign breast cancer detected':
                st.info("Symptoms")
                symptoms_text = "Early (benign) breast cancer symptoms may include a lump or thickening in the breast or underarm, changes in breast size or shape, dimpling or puckering of the skin, and nipple discharge (other than breast milk). The nipple may also turn inward, or the skin may become red, scaly, or swollen. While not all lumps are cancerous, it’s important to get any changes checked by a doctor."
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "The diagnosis of benign (non-cancerous) breast tumors involves a physical exam, imaging tests like mammograms or ultrasounds, and sometimes a biopsy (taking a small tissue sample for testing). Doctors check for lumps, changes in breast shape, or other symptoms. Benign tumors are usually harmless and don’t spread, but regular check-ups help ensure they stay that way."
                st.write(diagnosis_text)
                st.info("Treatment")
                treatment_text = "Benign breast tumors are non-cancerous growths that do not spread to other parts of the body. Treatment usually isn’t needed unless the lump is painful, growing, or causing discomfort. In such cases, doctors may recommend medication to reduce symptoms or a simple surgery to remove the lump. Regular check-ups and imaging tests help monitor any changes, ensuring the lump stays harmless."
                st.write(treatment_text)
                
            elif prediction == 'Malignant':
                st.info("Symptoms")
                symptoms_text = "Malignant breast cancer can cause a lump or thickening in the breast, changes in breast size or shape, and skin dimpling (like an orange peel). The nipple may turn inward, have discharge (sometimes bloody), or feel painful. The skin over the breast might become red, swollen, or flaky. Some people also experience underarm lumps or **constant breast pain. If you notice any of these signs, it's important to see a doctor." 
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "Malignant breast cancer is diagnosed through a few key steps. Doctors start with a physical exam to check for lumps or changes in the breast. A mammogram (breast X-ray) helps spot any unusual growths. If something looks suspicious, an ultrasound or MRI gives a clearer picture. To confirm cancer, a biopsy is done, where a small sample of tissue is taken and tested in a lab. Other tests, like blood work or genetic testing, may help determine the type and stage of cancer. Early diagnosis improves treatment options and outcomes."
                st.write(diagnosis_text)
                st.info("Treatment")
                treatment_text = "Malignant breast cancer is treated with a combination of methods depending on the stage and type. Surgery removes the tumor, while chemotherapy and radiation therapy help destroy cancer cells. Hormone therapy or targeted drugs may be used if the cancer is hormone-sensitive. In advanced cases, immunotherapy helps the body fight cancer. Early detection and the right treatment plan improve the chances of recovery."
                st.write(treatment_text)
            elif prediction == 'No tumor detected':  # Corrected here
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)

            else:
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)
    if prediction is not None:
        st.markdown("<h3 style='text-align: center;'>Thank you for using our application!</h3>", unsafe_allow_html=True)

        if st.button("Generate and Download Report"):
            # Create a PDF in memory
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Report Title
            c.setFont('Helvetica-Bold', 16)
            c.drawString(100, 750, "Kidney Tumor Detection Report")

            # Uploaded Image (if available)
            if uploaded_image is not None:
                try:
                    from PIL import Image  # Import PIL here

                    # Convert the OpenCV image to PIL format
                    pil_img = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
                    pil_img = Image.fromarray(pil_img)
                    pil_img = pil_img.convert("RGB")  # Ensure RGB color mode

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_img.save(tmp_file, "PNG")
                        temp_image_path = tmp_file.name

                    # Use the temporary file path with drawImage
                    c.drawImage(temp_image_path, 200, 590, width=200, height=150, mask='auto')
                    c.drawString(100, 570, "Uploaded Image:")

                except Exception as e:
                    c.setFont('Helvetica-Oblique', 10)
                    c.drawString(100, 580, f"Error embedding image: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            # Prediction Result
            c.setFont('Helvetica-Bold', 12)
            c.drawString(100, 550, f"Prediction: {prediction}")

            # Symptoms and Diagnosis
            style = styles["Normal"]
            style.leading = 14  # Adjust line spacing

            y_position = 530
            left_margin = 100
            text_width = 400  # Adjust as needed

            def add_paragraph(text, y_pos):
                p = Paragraph(text, style)
                p.wrapOn(c, text_width, 1000)
                p.drawOn(c, left_margin, y_pos - p.height)
                return y_pos - p.height - 10  # Update y_position


            if symptoms_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Symptoms:")
                y_position -= 15
                y_position = add_paragraph(symptoms_text, y_position)

            if diagnosis_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Diagnosis:")
                y_position -= 15
                y_position = add_paragraph(diagnosis_text, y_position)

            if additional_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Additional Information:")
                y_position -= 15
                y_position = add_paragraph(additional_text, y_position)
                
            if treatment_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Treatment:")
                y_position -= 15
                y_position = add_paragraph(treatment_text, y_position)



            c.save()
            buffer.seek(0)

            st.download_button(
                label="Download Report",
                data=buffer,
                file_name="kidney_tumor_report.pdf",
                mime="application/pdf",
            )
              
def show_kidneytumor():
    st.markdown(
    """
    <style>
        .stAlert {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center;'>Kidney Tumor Detection</h1>", unsafe_allow_html=True)
    st.warning(" Warning: Kindly ensure that only kidney tumor-related messages/files are uploaded. Thank you!", icon="⚠")
    st.warning(" Warning: after every use if you want to use again please reload web app. Thank you!", icon="⚠")
    classes = {"tumor_k": 0,"normal_k": 1}
    dec = {0: 'Tumor detected', 1: "No tumor detected" }
    button_style = f"""
        <style>
        div.stButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        div.stDownloadButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stDownloadButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
                 font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        </style>
        """
    st.markdown(button_style, unsafe_allow_html=True)


    def load_data(data_dir):
        x, y = [], []
        for cls in classes:
            pth = os.path.join(data_dir, cls)
            if os.listdir(pth):
                for img_name in os.listdir(pth):
                    try:
                        img = cv2.imread(os.path.join(pth, img_name), cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                            img = cv2.resize(img, (100, 100))
                            x.append(img)
                            y.append(classes[cls])
                    except Exception as e:
                        st.error(f"Error loading image {img_name}: {e}")
        return np.array(x), np.array(y)


    data_dir = "C:/Users/hp/Documents/braintumordetection/braintumordetection"  #changed the directory
    x, y = load_data(data_dir)


    x_flat = x.reshape(len(x), -1) / 255.0


    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42)


    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)


    lg = LogisticRegression(C=0.1)
    lg.fit(x_train_pca, y_train)
   
    st.write("Welcome to our MRI Detection web application! This innovative tool utilizes advanced machine learning algorithms to analyze MRI scans and detect the presence of tumors. Developed by a team of medical imaging experts and data scientists, our app offers a reliable and efficient solution for early diagnosis and treatment planning.")

    col1, col2 = st.columns((1, 1))
    prediction_made = False
    symptoms_text = None
    diagnosis_text = None
    additional_text = None
    treatment_text = None
    uploaded_image = None
    prediction = None

    with col1:
        
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:

            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            uploaded_image = img # Save uploaded image
            if img is not None:
                img_resized = cv2.resize(img, (100, 100))
                img_normalized = img_resized / 255.0
                img_input = img_normalized.reshape(1, -1)
                img_pca = pca.transform(img_input)
                prediction_lg = lg.predict(img_pca)


                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode()
                st.markdown(f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{img_base64}" alt="Uploaded Image" width="200"></div>',
                            unsafe_allow_html=True
                           )
                prediction = dec[prediction_lg[0]]
                st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)

    with col2:
        if prediction is not None:

            if prediction == 'Tumor detected':
                st.info("Symptoms")
                symptoms_text = "Symptoms of kidney tumors vary based on the type and size of the tumor, as well as the hormones affected. Common symptoms include: Headaches, Vision problems, especially if the tumor presses on the optic nerves, Nausea and vomiting, Hormonal imbalances that may lead to conditions like: Unexplained lactation in women, Reduced libido or fertility issues in men" #Replace with kidney tumor symtoms
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "To diagnose a kidney tumor, doctors usually start with imaging tests, like an ultrasound, CT scan, or MRI, to look for unusual growths in the kidneys. If something suspicious is found, a biopsy may be done, where a small piece of the tumor is removed and examined to check if it’s cancerous. Blood and urine tests may also be used to gather more information. Early detection is important for better treatment options." #Replace with kidney tumor diagnosis
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Treatment for a kidney tumor depends on its size and whether it has spread. Small tumors may be monitored with regular scans if they are not growing. Surgery is the most common treatment, where part or all of the kidney is removed. In some cases, freezing (cryotherapy) or heating (radiofrequency ablation) the tumor can destroy cancer cells without surgery. If the cancer has spread, targeted therapy, immunotherapy, or radiation may be used to slow its growth."
                st.write(treatment_text)

            elif prediction == 'No tumor detected':  # Corrected here
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)

            else:
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)
    if prediction is not None:
        st.markdown("<h3 style='text-align: center;'>Thank you for using our application!</h3>", unsafe_allow_html=True)

        if st.button("Generate and Download Report"):
            # Create a PDF in memory
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Report Title
            c.setFont('Helvetica-Bold', 16)
            c.drawString(100, 750, "Kidney Tumor Detection Report")

            # Uploaded Image (if available)
            if uploaded_image is not None:
                try:
                    from PIL import Image  # Import PIL here

                    # Convert the OpenCV image to PIL format
                    pil_img = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
                    pil_img = Image.fromarray(pil_img)
                    pil_img = pil_img.convert("RGB")  # Ensure RGB color mode

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_img.save(tmp_file, "PNG")
                        temp_image_path = tmp_file.name

                    # Use the temporary file path with drawImage
                    c.drawImage(temp_image_path, 200, 590, width=200, height=150, mask='auto')
                    c.drawString(100, 570, "Uploaded Image:")

                except Exception as e:
                    c.setFont('Helvetica-Oblique', 10)
                    c.drawString(100, 580, f"Error embedding image: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            # Prediction Result
            c.setFont('Helvetica-Bold', 12)
            c.drawString(100, 550, f"Prediction: {prediction}")

            # Symptoms and Diagnosis
            style = styles["Normal"]
            style.leading = 14  # Adjust line spacing

            y_position = 530
            left_margin = 100
            text_width = 400  # Adjust as needed

            def add_paragraph(text, y_pos):
                p = Paragraph(text, style)
                p.wrapOn(c, text_width, 1000)
                p.drawOn(c, left_margin, y_pos - p.height)
                return y_pos - p.height - 10  # Update y_position


            if symptoms_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Symptoms:")
                y_position -= 15
                y_position = add_paragraph(symptoms_text, y_position)

            if diagnosis_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Diagnosis:")
                y_position -= 15
                y_position = add_paragraph(diagnosis_text, y_position)

            if additional_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Additional Information:")
                y_position -= 15
                y_position = add_paragraph(additional_text, y_position)
                
            if treatment_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Treatment:")
                y_position -= 15
                y_position = add_paragraph(treatment_text, y_position)



            c.save()
            buffer.seek(0)

            st.download_button(
                label="Download Report",
                data=buffer,
                file_name="kidney_tumor_report.pdf",
                mime="application/pdf",
            )

            
              
def show_lungcancer():
    st.markdown(
    """
    <style>
        .stAlert {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center;'>Lung Cancer Detection</h1>", unsafe_allow_html=True)
    st.warning(" Warning: Kindly ensure that only Lung Cancer-related messages/files are uploaded. Thank you!", icon="⚠")
    st.warning(" Warning: after every use if you want to use again please reload web app. Thank you!", icon="⚠")
    classes = {"adeL": 0,"normalL": 1,"largL":2,"sqaL":3}
    dec = {0: 'Adenocarcinoma cancer detected', 1: "No cancer detected" ,2: 'Large cell carcinoma detected',3: 'Squamous cell cancer detected' }
    button_style = f"""
        <style>
        div.stButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        div.stDownloadButton {{
            text-align: center; /* Center the button horizontally */
        }}
        div.stDownloadButton > button:first-child {{
            background-color: #87CEEB;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
                 font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            /* Remove width: 100% */
        }}
        </style>
        """
    st.markdown(button_style, unsafe_allow_html=True)


    def load_data(data_dir):
        x, y = [], []
        for cls in classes:
            pth = os.path.join(data_dir, cls)
            if os.listdir(pth):
                for img_name in os.listdir(pth):
                    try:
                        img = cv2.imread(os.path.join(pth, img_name), cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                            img = cv2.resize(img, (100, 100))
                            x.append(img)
                            y.append(classes[cls])
                    except Exception as e:
                        st.error(f"Error loading image {img_name}: {e}")
        return np.array(x), np.array(y)


    data_dir = "C:/Users/hp/Documents/braintumordetection/braintumordetection"#changed the directory
    x, y = load_data(data_dir)


    x_flat = x.reshape(len(x), -1) / 255.0


    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42)


    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)


    lg = LogisticRegression(C=0.1)
    lg.fit(x_train_pca, y_train)
   
    st.write("Welcome to our MRI Detection web application! This innovative tool utilizes advanced machine learning algorithms to analyze MRI scans and detect the presence of tumors. Developed by a team of medical imaging experts and data scientists, our app offers a reliable and efficient solution for early diagnosis and treatment planning.")

    col1, col2 = st.columns((1, 1))
    prediction_made = False
    symptoms_text = None
    diagnosis_text = None
    additional_text = None
    treatment_text = None
    uploaded_image = None
    prediction = None

    with col1:
        
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:

            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            uploaded_image = img # Save uploaded image
            if img is not None:
                img_resized = cv2.resize(img, (100, 100))
                img_normalized = img_resized / 255.0
                img_input = img_normalized.reshape(1, -1)
                img_pca = pca.transform(img_input)
                prediction_lg = lg.predict(img_pca)


                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode()
                st.markdown(f'<div style="text-align: center;"><img src="data:image/jpeg;base64,{img_base64}" alt="Uploaded Image" width="200"></div>',
                            unsafe_allow_html=True
                           )
                prediction = dec[prediction_lg[0]]
                st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)

    with col2:
        if prediction is not None:

            if prediction == 'Adenocarcinoma cancer detected':
                st.info("Symptoms")
                symptoms_text = "Adenocarcinoma is a type of cancer that can start in different parts of the body, like the lungs, colon, or breast. Its symptoms can vary depending on where it begins, but common signs include unexplained weight loss, tiredness, pain in a specific area, and a lump or swelling. If it affects the lungs, a person might have a persistent cough or trouble breathing. If it’s in the digestive system, there may be stomach pain, changes in bowel habits, or blood in the stool. It’s important to see a doctor if any of these symptoms last for a while or get worse." #Replace with kidney tumor symtoms
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "To diagnose adenocarcinoma, doctors start by asking about your symptoms and medical history. Then they may do imaging tests like X-rays, CT scans, or MRIs to check for any unusual lumps or changes in your body. If they see something concerning, they will take a small sample of the tissue (called a biopsy) from the area and look at it under a microscope. This helps them confirm if the cells are cancerous and what type of cancer it is. Blood tests and other scans may also be done to see if the cancer has spread." #Replace with kidney tumor diagnosis
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Common treatments include surgery to remove the cancer, chemotherapy (medicine that kills cancer cells), radiation therapy (using high-energy rays to kill cancer), and sometimes newer treatments like targeted therapy or immunotherapy, which help the body fight the cancer more directly. The goal is to remove or destroy the cancer and help the person live longer and feel better."
                st.write(treatment_text)
                
            elif prediction == 'Large cell carcinoma detected':
                st.info("Symptoms")
                symptoms_text = "Large cell carcinoma is a fast-growing type of lung cancer. Symptoms may include a persistent cough, chest pain, coughing up blood, shortness of breath, feeling very tired, and unexplained weight loss. Some people may also have frequent lung infections like pneumonia." #Replace with kidney tumor symtoms
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "To diagnose, doctors usually start with imaging tests like chest X-rays or CT scans to look for anything unusual in the lungs. If something suspicious is found, they may do a biopsy—this means taking a small sample of lung tissue to check under a microscope for cancer cells. Blood tests and PET scans may also help see if the cancer has spread. Early detection can make a big difference in treatment." #Replace with kidney tumor diagnosis
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Doctors often use surgery to remove the tumor if it’s caught early. If it’s more advanced, treatment may include chemotherapy (strong medicines to kill cancer cells), radiation therapy (high-energy rays to shrink the cancer), or newer treatments like immunotherapy, which helps the body’s own immune system fight the cancer. The goal is to remove or control the cancer and help the person feel better."
                st.write(treatment_text)
                
            if prediction == 'Squamous cell cancer detected':
                st.info("Symptoms")
                symptoms_text = "Symptoms can vary from person to person but commonly include a persistent cough that doesn’t go away, chest pain, coughing up blood, shortness of breath, feeling fatigued or weak, and unexplained weight loss. People may also experience frequent lung infections or hoarseness." #Replace with kidney tumor symtoms
                st.write(symptoms_text)
                st.info("Diagnosis")
                diagnosis_text = "For diagnosis, doctors rely on several tests. A chest X-ray or CT scan may reveal abnormal areas in the lungs. If a suspicious spot is found, the doctor will likely perform a biopsy, either by inserting a needle into the lung or using a bronchoscope (a thin tube) to take a sample of lung tissue. Additional tests like PET scans or MRI scans might also be done to check if the cancer has spread to other areas of the body." #Replace with kidney tumor diagnosis
                st.write(diagnosis_text)
                
                st.info("Treatment")
                treatment_text = "Treatment depends on the stage of the cancer. If it’s found early and hasn’t spread, surgery might be an option to remove the tumor. For cancers that have spread or are harder to treat, chemotherapy, radiation therapy, or targeted therapies may be recommended. Immunotherapy, which boosts the body's immune system to fight the cancer, is also being used more frequently for squamous cell lung cancer. The treatment plan is often personalized to the patient’s condition and the cancer’s characteristics."
                st.write(treatment_text)
                

            elif prediction == 'No tumor detected':  # Corrected here
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)

            else:
                st.info("The image does not match any known tumor type.")
                additional_text = """
    The image you uploaded does not correspond to any known tumor type recognized by our machine learning model. This could be due to several reasons:

    1. Image Quality: The uploaded image may not have sufficient resolution or clarity for accurate analysis. High-quality images are essential for effective detection and diagnosis.

    2. Tumor Type: The model is trained on specific types of tumors, including brain tumors, breast cancer, and cervical cancer. If the tumor type present in the image is not among these, the model will not be able to provide a prediction.

    3. Non-Tumorous Conditions: The image may depict a benign condition or normal tissue that does not indicate the presence of a tumor. In such cases, it's important to consult with a medical professional for further evaluation.

    4. Model Limitations: Machine learning models are not infallible and have limitations. They may not always generalize well to unseen data or rare tumor types that were not included in the training dataset.

    If you believe that the image contains a tumor or if you have concerns about your health, we strongly recommend consulting a healthcare professional for a comprehensive evaluation and diagnosis. Early detection and accurate diagnosis are crucial for effectivepage1 treatment and better health outcomes.
    """
                st.write(additional_text)
    if prediction is not None:
        st.markdown("<h3 style='text-align: center;'>Thank you for using our application!</h3>", unsafe_allow_html=True)

        if st.button("Generate and Download Report"):
            # Create a PDF in memory
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Report Title
            c.setFont('Helvetica-Bold', 16)
            c.drawString(100, 750, "Kidney Tumor Detection Report")

            # Uploaded Image (if available)
            if uploaded_image is not None:
                try:
                    from PIL import Image  # Import PIL here

                    # Convert the OpenCV image to PIL format
                    pil_img = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
                    pil_img = Image.fromarray(pil_img)
                    pil_img = pil_img.convert("RGB")  # Ensure RGB color mode

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_img.save(tmp_file, "PNG")
                        temp_image_path = tmp_file.name

                    # Use the temporary file path with drawImage
                    c.drawImage(temp_image_path, 200, 590, width=200, height=150, mask='auto')
                    c.drawString(100, 570, "Uploaded Image:")

                except Exception as e:
                    c.setFont('Helvetica-Oblique', 10)
                    c.drawString(100, 580, f"Error embedding image: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            # Prediction Result
            c.setFont('Helvetica-Bold', 12)
            c.drawString(100, 550, f"Prediction: {prediction}")

            # Symptoms and Diagnosis
            style = styles["Normal"]
            style.leading = 14  # Adjust line spacing

            y_position = 530
            left_margin = 100
            text_width = 400  # Adjust as needed

            def add_paragraph(text, y_pos):
                p = Paragraph(text, style)
                p.wrapOn(c, text_width, 1000)
                p.drawOn(c, left_margin, y_pos - p.height)
                return y_pos - p.height - 10  # Update y_position


            if symptoms_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Symptoms:")
                y_position -= 15
                y_position = add_paragraph(symptoms_text, y_position)

            if diagnosis_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Diagnosis:")
                y_position -= 15
                y_position = add_paragraph(diagnosis_text, y_position)

            if additional_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Additional Information:")
                y_position -= 15
                y_position = add_paragraph(additional_text, y_position)
                
            if treatment_text:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(100, y_position, "Treatment:")
                y_position -= 15
                y_position = add_paragraph(treatment_text, y_position)



            c.save()
            buffer.seek(0)

            st.download_button(
                label="Download Report",
                data=buffer,
                file_name="kidney_tumor_report.pdf",
                mime="application/pdf",
            )
            
def main():
    st.set_page_config(layout="wide")
    set_bg_style()

    # Sidebar navigation
    #st.sidebar.title("option")
    #page = st.sidebar.radio("", ["Home", "Brain Tumor Detection"])
    #orientation="horizontal"
    selected=option_menu(
        menu_title="DETECT WELL",
        options=["Home", "Brain Tumor","Breast Cancer","Kidney Tumor","Lung Cancer"],
        icons=["house", "plus-circle", "gender-female", "droplet","lungs"],
        menu_icon="cast",
        default_index=0, # Corrected typo here
        orientation="horizontal",
       styles={
        "container": {"padding": "5px", "background-color": "white"},
        "menu-title": {"color": "black", "font-size": "20px", "font-weight": "bold"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px", "color": "black"},
        "nav-link-selected": {"background-color": "#87CEEB", "color": "white"},
    }
    )
    

    if selected == "Home":  # Changed 'page' to 'selected'
        show_home()
    elif selected == "Brain Tumor": # Changed 'page' to 'selected'
        

        show_braintumor()
        
      
    elif selected == "Kidney Tumor": # Changed 'page' to 'selected'
        
        

        show_kidneytumor()
    elif selected == "Breast Cancer": # Changed 'page' to 'selected'
        

        show_brestcancer()
        
    elif selected == "Lung Cancer": # Changed 'page' to 'selected'
        

        show_lungcancer()
        

   


        

if __name__ == "__main__":
    main()