import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av


@st.cache_resource
def load_model():
    return tf.saved_model.load('fruit model')

# Add title and favicon
st.set_page_config(page_title="Quality Control Food Raw Materials", page_icon="🍏")

# Adding CSS style
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://miro.medium.com/v2/resize:fit:720/format:webp/0*Ciet3UBlRwGcz7Sx");
        background-size: cover;
        background-attachment: fixed;
        background-color: #00325B;
        primaryColor: #FF8C02;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load konfigurasi dari file YAML
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize the authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# Render the login / register module
if not st.session_state.get('authentication_status'):
    tab_login, tab_register = st.tabs(['Login', 'Daftar Akun Baru'])
    with tab_login:
        name, authentication_status, username = authenticator.login('main')
    with tab_register:
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = (
                authenticator.register_user(pre_authorization=False, location='main')
            )
            if email_of_registered_user:
                st.success(
                    f'Registrasi berhasil untuk {name_of_registered_user} '
                    f'({email_of_registered_user}). Silakan login.'
                )
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
else:
    name, authentication_status, username = authenticator.login('main')

if authentication_status:
    # Display logout button
    st.sidebar.title(f'Welcome, {name}!')

    def preprocess_image(image):
        image = tf.image.resize(image, (256, 256))
        image = image.numpy().astype('float32')
        image = preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        return image

    def preprocess_webcam_image(image):
        return preprocess_image(image)

    def preprocess_uploaded_image(uploaded_file):
        image = tf.image.decode_image(uploaded_file.getvalue(), channels=3, expand_animations=False)
        return preprocess_image(image)

    # Function to make predictions
    def predict_image(image):
        prediction = model(image)
        return float(prediction.numpy()[0][0])

    # Load the SavedModel
    model = load_model()

    # Page title and subtitle
    st.title('🍏🍓🍌 Quality Control Food Raw Materials 🍍🥝🍇')
    st.markdown('Aplikasi ini dibuat untuk memprediksi kesegaran buah. Anda dapat mengunggah gambar buah atau menggunakan webcam untuk memperoleh prediksi. Buah yang diprediksi dapat berupa segar atau busuk. Aplikasi ini menggunakan model machine learning yang telah dilatih sebelumnya untuk memprediksi kesegaran buah.')

    # Menampilkan gambar contoh buah segar dan tidak segar
    st.subheader("Contoh Bahan baku yang lolos QC dan tidak lolos QC")
    col1, col2 = st.columns(2)
    with col1:
        st.image('fresh-orange-fruit.jpg', caption='Contoh Bahan Baku yang lolos QC', use_column_width=True)
    with col2:
        st.image('istockphoto-902552216-612x612.jpg', caption='Contoh Bahan baku yang tidak lolos QC', use_column_width=True)

    st.subheader('Silakan pilih opsi di sidebar untuk memilih sumber gambar (unggah gambar atau gunakan webcam).')

    # Sidebar option to select source
    option = st.sidebar.radio('Select an option:', ('Upload Image', 'Use Webcam'))

    # Reminder to clear uploaded image if switching to webcam
    if option == 'Upload Image':
        st.sidebar.warning('Jangan lupa untuk menghapus gambar yang diunggah sebelum menggunakan webcam!')

    if option == 'Use Webcam':
        st.markdown('Jika probabilitas lebih dari 50% maka buah tersebut sudah tidak layak untuk dikonsumsi.(Web cam akan membutuhkan sedikit waktu sebelum bisa memprediksi)')

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.prediction = None
                
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Preprocess image
                image = preprocess_webcam_image(img_rgb)
                
                # Make prediction
                self.prediction = predict_image(image)

                return av.VideoFrame.from_ndarray(img, format='bgr24')
        
        ctx = webrtc_streamer(
            key="example", 
            mode=WebRtcMode.SENDRECV, 
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), 
            video_processor_factory=VideoProcessor
        )

        if ctx.video_processor:
            prediction = ctx.video_processor.prediction
            if prediction is not None:
                prediction_text = 'Fresh' if prediction < 0.5 else 'Rotten'
                probability_text = f'Probability: {prediction * 100:.2f}%'
                st.write(f'Prediction: {prediction_text}')
                st.write(probability_text)
    
    else:
        # File uploader
        uploaded_file = st.sidebar.file_uploader('Pilih Gambarnya...', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = preprocess_uploaded_image(uploaded_file)
            prediction = predict_image(image)

            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            # Display prediction result
            st.subheader('Prediction:')
            if prediction < 0.5:
                st.success('Segar')
                st.write('Buah ini terlihat segar dan siap untuk dimakan!')
            else:
                st.error('Busuk')
                st.write('Oops! Buah ini tampaknya busuk. Sebaiknya dibuang.')

            # Display the prediction result using a pie chart
            st.subheader('Prediction Visualization:')
            labels = ['Busuk', 'Segar']
            sizes = [prediction, 1 - prediction]
            colors = ['#ff6961', '#77dd77']
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Additional information and tips
            if prediction < 0.5:
                st.info('Jika buah terlihat segar namun memiliki bau yang aneh, sebaiknya jangan dikonsumsi.')

                st.subheader('Informasi Tambahan:')
                st.write('### Tips Menjaga Buah Tetap Segar')
                st.write('- Simpan di Lemari Es: Suhu rendah memperlambat aktivitas mikroorganisme dan enzim, sehingga memperpanjang kesegaran buah.')
                st.write('- Pisahkan Buah: Beberapa buah menghasilkan gas etilen (seperti apel, pisang, dan tomat) yang dapat mempercepat pematangan buah lain. Simpan buah ini terpisah untuk mencegah pematangan cepat.')
                st.write('- Cuci dan Keringkan Buah: Sebelum menyimpan buah, cuci dan keringkan mereka untuk mengurangi mikroorganisme yang mungkin ada di permukaan.')
                st.write('- Gunakan Wadah Penyimpanan yang Tepat: Simpan buah dalam wadah tertutup atau bungkus plastik untuk mengurangi paparan udara dan mencegah oksidasi.')
                st.write('- Periksa Buah Secara Berkala: Periksa buah secara berkala dan pisahkan buah yang mulai membusuk untuk mencegah penyebaran ke buah lain.')
                st.write('- Simpan di Tempat yang Kering: Simpan buah di tempat yang kering untuk mencegah pertumbuhan jamur.')
                st.write('- Hindari Cidera Fisik: Tangani buah dengan hati-hati untuk menghindari memar atau luka yang dapat mempercepat pembusukan.')
            else:
                st.info('Pastikan untuk membuang buah yang terlihat busuk atau berjamur.')

                # Additional information for rotten fruit
                st.subheader('Informasi tentang Buah Busuk:')
                st.write('### Penyebab Buah Busuk')
                st.write('- Mikroorganisme (Bakteri dan Jamur): Bakteri dan jamur dapat tumbuh pada buah, terutama pada kondisi yang lembab dan hangat. Mereka menguraikan jaringan buah, menyebabkan pembusukan.')
                st.write('- Paparan Udara (Oksidasi): Ketika buah terpapar udara, proses oksidasi dapat terjadi, menyebabkan buah berubah warna, tekstur, dan rasa.')
                st.write('- Enzim Buah: Buah mengandung enzim yang memecah jaringan buah seiring waktu, terutama setelah buah dipetik.')
                st.write('- Suhu Tinggi: Suhu tinggi mempercepat aktivitas mikroorganisme dan enzim, sehingga mempercepat pembusukan buah.')
                st.write('- Kerusakan Fisik: Buah yang terluka atau memar lebih rentan terhadap serangan mikroorganisme dan pembusukan.')

                st.write('### Tanda-tanda Buah Busuk')
                st.write('- Perubahan Warna: Buah yang busuk biasanya berubah warna, misalnya menjadi cokelat atau hitam.')
                st.write('- Tekstur Lembek: Buah yang busuk akan terasa lembek dan berair.')
                st.write('- Bau Tidak Sedap: Buah yang busuk seringkali memiliki bau yang tidak sedap atau asam.')
                st.write('- Pertumbuhan Jamur: Buah yang busuk sering kali memiliki pertumbuhan jamur yang terlihat sebagai bercak putih, hijau, atau hitam.')

    # Display logout button
    authenticator.logout('Logout', 'sidebar')

    # Add footer
    st.markdown(
        """
        <style>
        .footer {
            display: block;
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #FF7E38;
            text-color: #FFFFFF;
            padding: 10px 0;
            text-align: center;
        }
        </style>
        <div class="footer">
            <p>Made with ❤️ by kelompok 36</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Handle authentication status
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
