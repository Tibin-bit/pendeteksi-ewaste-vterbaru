import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global AI E-Waste Detector Pro (Offline AI)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# ---------------------------------------------------------
# 2. INI SIALISASI MODEL VISION AI (ON-DEVICE / FAST INFERENCE)
# ---------------------------------------------------------
@st.cache_resource
def load_vision_model():
    """Memuat model Vision AI ringan & cepat tanpa butuh API Key."""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess, categories

model, preprocess, categories = load_vision_model()

# ---------------------------------------------------------
# 3. KNOWLEDGE BASE STANDAR UN E-WASTE MONITOR
# ---------------------------------------------------------
EWASTE_DATABASE = {
    "small_it": {
        "kategori_un": "6. Small IT & Telecommunication Equipment",
        "keywords": ["cellular telephone", "hand-held computer", "ipod", "laptop", "notebook", "computer keyboard", "mouse", "modem", "printer", "hard disc", "joystick"],
        "tingkat_bahaya": "Tinggi",
        "skor_bahaya": 8,
        "bahan_berbahaya": ["Timbal (Solder)", "Raksa (Layar)", "Kadmium (Baterai)", "Barium"],
        "potensi_logam": {"Emas (Au)": "Tinggi", "Perak (Ag)": "Sedang", "Tembaga (Cu)": "Tinggi"},
        "daur_ulang": 85,
        "instruksi": [
            "Lepaskan baterai lithium-ion secara terpisah dan simpan di wadah kering.",
            "Jangan merusak sirkuit papan (PCB) untuk mencegah pelepasan logam berat.",
            "Bawa ke e-waste collection center terdekat untuk ekstraksi logam mulia."
        ]
    },
    "screens": {
        "kategori_un": "2. Screens & Monitors",
        "keywords": ["television", "monitor", "screen", "crt", "flat panel"],
        "tingkat_bahaya": "Tinggi",
        "skor_bahaya": 9,
        "bahan_berbahaya": ["Raksa (Backlight)", "Timbal (Kaca CRT)", "Zat Phosphor"],
        "potensi_logam": {"Emas (Au)": "Sedang", "Perak (Ag)": "Rendah", "Tembaga (Cu)": "Tinggi"},
        "daur_ulang": 70,
        "instruksi": [
            "Hindari memecahkan panel layar kaca agar gas berbahaya tidak terhirup.",
            "Pisahkan kabel daya dan braket dudukan berbahan logam.",
            "Serahkan ke fasilitas fasilitas daur ulang berlisensi khusus penanganan raksa."
        ]
    },
    "temperature": {
        "kategori_un": "1. Temperature Exchange Equipment",
        "keywords": ["refrigerator", "ice maker", "air conditioner", "freezer"],
        "tingkat_bahaya": "Sangat Tinggi",
        "skor_bahaya": 9,
        "bahan_berbahaya": ["Gas Freon/CFC/HCFC (Ozon)", "Minyak Kompresor", "Ammonia"],
        "potensi_logam": {"Emas (Au)": "Tidak Ada", "Perak (Ag)": "Rendah", "Tembaga (Cu)": "Sangat Tinggi"},
        "daur_ulang": 90,
        "instruksi": [
            "Dilarang memotong pipa pendingin kompresor secara mandiri.",
            "Pastikan penanganan dilakukan oleh teknisi resmi untuk penyedotan gas pendingin.",
            "Besi kompresor dan pipa tembaga dapat didaur ulang secara maksimal."
        ]
    },
    "large_equip": {
        "kategori_un": "4. Large Equipment",
        "keywords": ["washer", "dishwasher", "stovetop", "microwave oven", "vacuum cleaner"],
        "tingkat_bahaya": "Sedang",
        "skor_bahaya": 6,
        "bahan_berbahaya": ["Kapasitor Minyak PCB", "Logam Berat Solder", "Plastik BFR"],
        "potensi_logam": {"Emas (Au)": "Rendah", "Perak (Ag)": "Rendah", "Tembaga (Cu)": "Sangat Tinggi"},
        "daur_ulang": 80,
        "instruksi": [
            "Lepaskan motor listrik internal untuk daur ulang tembaga dan besi murni.",
            "Pisahkan bagian bodi logam utama dari komponen plastik sintetis.",
            "Kirim motor dan sasis ke tempat penampungan besi tua/logam."
        ]
    },
    "small_equip": {
        "kategori_un": "5. Small Equipment",
        "keywords": ["camera", "reflex camera", "toaster", "waffle iron", "hair dryer", "electric fan", "iron", "blender", "coffee mug", "loudspeaker", "headphone"],
        "tingkat_bahaya": "Sedang",
        "skor_bahaya": 5,
        "bahan_berbahaya": ["BFR (Brominated Flame Retardants)", "Timbal", "Nikel"],
        "potensi_logam": {"Emas (Au)": "Rendah", "Perak (Ag)": "Sedang", "Tembaga (Cu)": "Sedang"},
        "daur_ulang": 75,
        "instruksi": [
            "Gulung kabel daya berbahan tembaga secara terpisah.",
            "Jika memiliki baterai tanam, pisahkan baterai sebelum dibuang.",
            "Komponen motor kecil dan kabel dapat disetorkan ke bank sampah lokal."
        ]
    }
}

def classify_ewaste(image):
    """Fungsi klasifikasi lokal serba cepat tanpa koneksi API eksternal."""
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 3)
    
    detected_label = categories[top_catid[0].item()]
    confidence = top_prob[0].item() * 100
    
    # Cari kecocokan dengan Knowledge Base UN E-Waste
    detected_key = "small_equip"  # Default fallback
    for key, data in EWASTE_DATABASE.items():
        for kw in data["keywords"]:
            if kw.lower() in detected_label.lower():
                detected_key = key
                break
                
    info = EWASTE_DATABASE[detected_key]
    
    formatted_name = detected_label.replace("_", " ").title()
    
    return {
        "nama_objek": formatted_name,
        "kategori_un": info["kategori_un"],
        "confidence": round(confidence, 1),
        "tingkat_bahaya": info["tingkat_bahaya"],
        "skor_bahaya": info["skor_bahaya"],
        "bahan_berbahaya": info["bahan_berbahaya"],
        "potensi_logam_mulia": info["potensi_logam"],
        "dapat_didaur_ulang_persen": info["daur_ulang"],
        "instruksi_penanganan": info["instruksi"]
    }

# ---------------------------------------------------------
# 4. SIDEBAR INFORMASI
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ AI Engine Status")
    st.success("🟢 **Vision AI Engine:** Active (Offline)")
    st.info("⚡ **Kecepatan Analisis:** < 0.5 Detik\n🔑 **API Key:** Tidak Diperlukan")
    st.markdown("---")
    st.markdown("### 📋 Standar Klasifikasi")
    st.caption("Berpedoman pada standar **UN Global E-Waste Monitor** (United Nations University / UNITAR).")

# ---------------------------------------------------------
# 5. TAMPILAN UTAMA APLIKASI
# ---------------------------------------------------------
st.title("⚡ Global AI E-Waste Detector Pro")
st.markdown("Sistem Pengenal & Analisis Bahaya Sampah Elektronik Berbasis Vision AI Cepat")

tab1, tab2 = st.tabs(["🔍 Analisis E-Waste", "📊 Dashboard & Riwayat"])

# --- TAB 1: ANALISIS GAMBAR ---
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="medium")
    
    with col_input:
        st.subheader("1. Pilih Sumber Gambar")
        source = st.radio("Metode Input:", ["Kamera Langsung 📷", "Unggah Berkas 📁"], horizontal=True)
        
        input_image = None
        if "Kamera" in source:
            cam_file = st.camera_input("Ambil Foto Perangkat E-Waste")
            if cam_file:
                input_image = Image.open(cam_file).convert("RGB")
        else:
            uploaded_file = st.file_uploader("Pilih gambar perangkat (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                input_image = Image.open(uploaded_file).convert("RGB")
                
        if input_image:
            st.image(input_image, caption="Gambar Siap Dianalisis", use_container_width=True)
            analyze_btn = st.button("🚀 Jalankan Analisis AI Kilat", type="primary", use_container_width=True)

    with col_output:
        st.subheader("2. Hasil Deteksi & Analisis Mendalam")
        
        if 'analyze_btn' in locals() and analyze_btn:
            if input_image is None:
                st.warning("⚠️ **Gambar Belum Ada!** Ambil foto atau unggah gambar e-waste terlebih dahulu.")
            else:
                with st.spinner("⚡ Menganalisis dengan Vision AI Engine..."):
                    data = classify_ewaste(input_image)
                    
                    st.session_state.detection_history.append({
                        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "nama": data["nama_objek"],
                        "kategori": data["kategori_un"],
                        "bahaya": data["tingkat_bahaya"],
                        "daur_ulang": data["dapat_didaur_ulang_persen"]
                    })
                    
                    st.success(f"✅ **Analisis Selesai!** (Tingkat Akurasi Deteksi: `{data['confidence']}%`)")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Perangkat Terdeteksi", data["nama_objek"])
                    m2.metric("Tingkat Bahaya", data["tingkat_bahaya"], delta=f"Skor {data['skor_bahaya']}/10", delta_color="inverse")
                    m3.metric("Potensi Daur Ulang", f"{data['dapat_didaur_ulang_persen']}%")
                    
                    st.markdown("---")
                    st.markdown(f"**📂 Kategori UN E-Waste:** `{data['kategori_un']}`")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("🚨 **Bahan / Zat Berbahaya:**")
                        for bahan in data["bahan_berbahaya"]:
                            st.write(f"- {bahan}")
                            
                    with col_b:
                        st.markdown("💎 **Potensi Logam Mulia:**")
                        for k, v in data["potensi_logam_mulia"].items():
                            st.write(f"- **{k}:** {v}")
                            
                    st.markdown("---")
                    st.markdown("🛠️ **Instruksi Penanganan & Daur Ulang Aman:**")
                    for idx, step in enumerate(data["instruksi_penanganan"], 1):
                        st.write(f"**{idx}.** {step}")

# --- TAB 2: DASHBOARD & RIWAYAT ---
with tab2:
    st.subheader("📊 Rekapitulasi Deteksi E-Waste")
    
    if len(st.session_state.detection_history) > 0:
        df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df, names="kategori", title="Distribusi Kategori UN E-Waste", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            fig_bar = px.bar(df, x="nama", y="daur_ulang", color="bahaya", title="Persentase Daur Ulang per Objek")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Belum ada riwayat deteksi pada sesi ini. Lakukan deteksi di Tab 1 untuk melihat dashboard.")
