import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                        ForgotError,
                                                        LoginError,
                                                        RegisterError,
                                                        ResetError,
                                                        UpdateError) 
from PIL import Image
import numpy as np
from googleapiclient.discovery import build

import tensorflow as tf
import tensorflow_hub as hub


# Fungsi untuk memuat model dari TensorFlow Hub
@st.cache(allow_output_mutation=True)
def load_model():
    model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
    return model

# Fungsi untuk memuat label
@st.cache_data
def load_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()
    return labels

# Fungsi untuk mempreproses gambar
def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (224, 224)) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

# Fungsi untuk memprediksi kelas dari gambar
def predict(image, model, labels):
    image = preprocess_image(image)
    predictions = model(image)
    predicted_class = np.argmax(predictions[0], axis=-1)
    return labels[predicted_class]

# Fungsi utama untuk aplikasi deteksi makanan
def deteksi_makanan():
    # Memuat model dan label
    model = load_model()
    labels = load_labels()
    
    # Judul aplikasi
    st.title("Aplikasi Deteksi Makanan")
    st.write("Unggah gambar dan model akan memprediksi jenis makanan tersebut.")
    
    # Mengunggah file
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # image = Image.open(uploaded_file)
        # st.image(image, caption='Gambar yang diunggah', use_column_width=True)
        # st.write("")
        # st.write("Sedang mengklasifikasikan...")
        
        # # Prediksi label gambar
        # label = predict(image, model, labels)
        # st.write(f"Prediksi: {label}")
        # Membuat layout dengan dua kolom
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Sedang mengklasifikasikan...")
            
            # Prediksi label gambar
            image = Image.open(uploaded_file)
            label = predict(image, model, labels)
            st.write("Prediksi:")
            st.title(f"{label}")
            
        with col2:
            st.image(image, caption='Gambar yang diunggah', use_column_width=True)


# Memuat konfigurasi di luar fungsi main
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# Fungsi untuk halaman registrasi
def register():
    st.title("Form Register")
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
        if email_of_registered_user:
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)
    if st.button("Kembali"):
        st.session_state["page"] = "login"

# Fungsi untuk halaman lupa password
def forgot_password():
    st.title("Form Lupa Password")
    try:
        username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password()
        if username_of_forgotten_password:
            st.success('New password will be sent securely')
        elif username_of_forgotten_password == False:
            st.error('Username not found')
    except Exception as e:
        st.error(e)
    if st.button("Kembali"):
        st.session_state["page"] = "login"

# Fungsi untuk halaman lupa username
def forgot_username():
    st.title("Form Lupa Username")
    try:
        username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username()
        if username_of_forgotten_username:
            st.success('Username to be sent securely')
            st.session_state["page"] = "login"
            # The developer should securely transfer the username to the user.
        elif username_of_forgotten_username == False:
            st.error('Email not found')
    except Exception as e:
        st.error(e)
    if st.button("Kembali"):
        st.session_state["page"] = "login"

def bmi_calculator():
    st.title("BMI Calculator")
    st.write("Indeks Massa Tubuh (BMI) adalah ukuran yang menggunakan tinggi dan berat badan Anda untuk menentukan apakah berat badan Anda sehat.")
    st.write("""
    - Kurang berat badan: BMI < 18.5
    - Berat badan normal: BMI = 18.5–24.9
    - Kelebihan berat badan: BMI = 25–29.9
    - Obesitas: BMI >= 30
    """)
    
    gender = st.radio("Jenis Kelamin", ("Pria", "Wanita"))
    height = st.number_input("Tinggi badan (cm)", min_value=0.0, step=0.1)
    weight = st.number_input("Berat Badan (kg)", min_value=0.0, step=0.1)
    
    if st.button("Hitung BMI"):
        if height > 0 and weight > 0:
            height_m = height / 100  # Convert height to meters
            bmi = weight / (height_m ** 2)
            st.success(f"BMI Anda adalah {bmi:.2f}")
            
            if bmi < 18.5:
                st.write("Anda memiliki berat badan kurang. Perhatikan pola makan dan gizi Anda.")
            elif 18.5 <= bmi <= 24.9:
                st.write("Anda memiliki berat badan normal. Tetap jaga gaya hidup sehat!")
            elif 25 <= bmi <= 29.9:
                st.write("Anda memiliki kelebihan berat badan. Pertimbangkan pola makan dan aktivitas fisik Anda.")
            else:
                st.write("Anda memiliki obesitas. Segera konsultasikan dengan dokter atau ahli gizi.")
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            categories = ['Kurus', 'Normal', 'Gemuk', 'Obesitas']
            colors = ['blue', 'green', 'orange', 'red']
            ax.barh(categories, [18.5, 24.9, 29.9, 35], color=colors, alpha=0.6)
            ax.axvline(bmi, color='black', linestyle='--')
            st.pyplot(fig)
        else:
            st.error("Tinggi dan berat badan harus lebih dari 0 untuk menghitung BMI.")

def youtube_search(query):
    # Replace with your YouTube API key
    api_key = "AIzaSyCpD47FNPyg2ZlmD_jj-0NwnivEBIT-9Ho"
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=1
    )
    response = request.execute()
    
    if response['items']:
        video_id = response['items'][0]['id']['videoId']
        return f"https://www.youtube.com/watch?v={video_id}"
    else:
        return None

def video_search():
    st.title("Cari Video YouTube")
    query = st.text_input("Masukkan kata kunci pencarian video:")
    
    if st.button("Cari"):
        video_url = youtube_search(query)
        if video_url:
            st.video(video_url)
        else:
            st.error("Video tidak ditemukan")

def setting_account():
    st.title("Pengaturan Akun")
    option = st.radio("Pilih opsi pengaturan:", ("Ubah Nama Pengguna", "Ubah Kata Sandi"))
    
    if option == "Ubah Nama Pengguna":
        new_username = st.text_input("Masukkan nama pengguna baru:")
        if st.button("Simpan"):
            # Logika untuk menyimpan nama pengguna baru ke dalam konfigurasi
            config['username'] = new_username
            st.success("Nama pengguna berhasil diubah.")
    
    elif option == "Ubah Kata Sandi":
        new_password = st.text_input("Masukkan kata sandi baru:", type="password")
        confirm_password = st.text_input("Konfirmasi kata sandi baru:", type="password")
        if st.button("Simpan"):
            # Logika untuk menyimpan kata sandi baru ke dalam konfigurasi
            if new_password == confirm_password:
                config['password'] = new_password
                st.success("Kata sandi berhasil diubah.")
            else:
                st.error("Kata sandi dan konfirmasi tidak cocok.")

def resep():
    st.title("Resep")
    # Contoh data resep
    resep_data = [
        {"nama": "1.	QUINOA SALAD TROPIS", "foto": "https://i.ibb.co.com/d09js6w/SALAD.jpg", "caption": """Bahan Mentah
• 2 Avocados, large
• 3 packed cups Baby spinach
• 3 Limes, medium
• 2 Mangos, large
• 1/2 cup Quinoa
• 1/4 tsp Black pepper, freshly ground
• 1/4 cup Olive oil
• 1/4 cup Almonds
Bahan Cair
• 1 cup Water 
Cara pembuatan : campurkan semua
"""}, {"nama": "2.	SALAD BUAH SEGAR", "foto": "https://i.ibb.co.com/1qhctjm/SALAD-BUAH.jpg", "caption": """Bahan-bahan
 30 menit
 5 porsi
1 kg buah semangka
1/2 kg melon
2 apel
2 pir
1 genggam murbey
1/2 kg mayonais
2 saschet skm putih
3 sdm gula pasir
1 botol kecil cimory blubbery
Cara Membuat
1.	Cuci dan potong-potong semua buah kecuali murbey
2.	Campurkan mayonais, gula, skm, cimory jika kurang manis bisa di tambah gula
3.	Campur buah dengan mayonais secukupnya..siap dihidangkan
"""}, {"nama": "3.	BROKOLI SAUS TIRAM  ", "foto": "https://i.ibb.co.com/bH4jQxT/Korean-Roled.jpg", "caption": """Bahan-bahan
1 bonggol Brokoly
1/2 bh bwg bombay
4 siung bwg putih
2 bh telor
Daun pisang / plastic wrap lebih bagus double
Bumbu saus :
2 sdm saos tiram
2 sdm kecap asin
1 sdt kaldu jamur
1 Sdm gula pasir
1 sdt kecap manis
50 ml air
Cara Pembuatan : 
1.	Kocok telor + 1 sdt garam + air 100 ml, saring dan masukan ke mangkuk tahan panas dg menutupnya daun pisang/plastic wrap, kukus hga 30mnt.
2.	Potong brokoly & batangnya. Rendam dg 1 sdm garam 5mnt.
3.	Cincang bwg putih & iris tipis bwg bombay. Cincangan bwg putih di bilas dulu dg air mengalir menggunakan saringan dan peras & tiriskan. Siapkan sausnya dg mencampur semua bumbu saus dlm 1 wadah. Siapkan campuran 1 sdm maizena & air di wadah tersendiri. Tumis bwg putih cincang dg minyak. Angkat tiriskan.
4.	Masukan larutan saos da aduk hga mendidih. (Bisa tes rasa) Jika dirasa sudah pas masukan larutan maizena.terakhir Sajikan telur kukus siram brokoly saus Tiram & taburan bwg goreng.
"""},]
    
    for resep in resep_data:
        st.header(resep["nama"])
        st.image(resep["foto"], write=resep["caption"], use_column_width=True)

def main():
    st.cache()
    
    # Tentukan halaman yang akan ditampilkan berdasarkan tombol yang ditekan
    if "page" not in st.session_state:
        st.session_state["page"] = "login"
        
        
    if st.session_state["authentication_status"]:
        authenticator.logout("Logout", "main")
        st.title('Calor Mine')
        st.write(f'Hallo *{st.session_state["name"]}*')
        
        # Menu after login using horizontal layout
        menu = ["Deteksi Makanan", "BMI", "Video", "Resep", "Setting Account"]
        cols = st.columns(len(menu))
        for i, col in enumerate(cols):
            if col.button(menu[i]):
                st.session_state["menu"] = menu[i]
        
        # Show different content based on selected menu
        selected_menu = st.session_state.get("menu", "Deteksi Makanan")
        if selected_menu == "Deteksi Makanan":
            deteksi_makanan()  
        elif selected_menu == "BMI":
            bmi_calculator()
        elif selected_menu == "Video":
            video_search()
        elif selected_menu == "Resep":
            resep()
        elif selected_menu == "Setting Account":
            setting_account()
    
    else:
        if st.session_state["page"] == "register":
            register()
        elif st.session_state["page"] == "forgot_password":
            forgot_password()
        elif st.session_state["page"] == "forgot_username":
            forgot_username()
        else:
            # Halaman Login
            if st.session_state["authentication_status"] is False:
                st.error('Username/password is incorrect')
            elif st.session_state["authentication_status"] is None:
                st.warning('Please enter your username and password')
            
            # Form login
            authenticator.login()
            
            # Tombol navigasi vertikal
            st.button("Register", on_click=lambda: st.session_state.update({"page": "register"}))
            st.button("Lupa Password", on_click=lambda: st.session_state.update({"page": "forgot_password"}))
            st.button("Lupa Username", on_click=lambda: st.session_state.update({"page": "forgot_username"}))
    
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

# Jalankan aplikasi
if __name__ == "__main__":
    main()
