"""
Multilingual Agronomic Advisory Bundles for Indian Soybean Farmers and Extension Workers.
Supports English (en), Hindi (hi), and Marathi (mr).
"""
from typing import Any, Dict

ADVISORIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Rust": {
        "en": {
            "name": "Asian Soybean Rust (Phakopsora pachyrhizi)",
            "urgency": "CRITICAL",
            "symptoms": "Minute reddish-brown to tan pustules on leaf undersides causing premature defoliation.",
            "organic": "Spray neem oil 5ml/liter or 5% NSKE (Neem Seed Kernel Extract) at first notice.",
            "chemical": "Spray Tebuconazole 25.9% EC @ 250-300 ml/acre or Hexaconazole 5% EC @ 400 ml/acre in 200 liters of water.",
        },
        "hi": {
            "name": "सोयाबीन रतुआ / किट्ट रोग (Rust)",
            "urgency": "अत्यंत गंभीर (CRITICAL)",
            "symptoms": "पत्तियों की निचली सतह पर छोटे भूरे या लाल रंग के दाने/फफोले बनते हैं और पत्तियां झड़ने लगती हैं।",
            "organic": "शुरुआती लक्षण पर नीम का तेल 5 मिली/लीटर या 5% नीम गिरी अर्क (NSKE) का छिड़काव करें।",
            "chemical": "टेबुकोनाज़ोल 25.9% EC (250-300 मिली/एकड़) या हेक्साकोनाज़ोल 5% EC (400 मिली/एकड़) का 200 लीटर पानी में छिड़काव करें।",
        },
        "mr": {
            "name": "सोयाबीन तांबेरा रोग (Rust)",
            "urgency": "अत्यंत गंभीर (CRITICAL)",
            "symptoms": "पानांच्या खालच्या बाजूवर लालसर-तपकिरी रंगाचे लहान पुरळ येतात आणि पाने अकाली गळतात.",
            "organic": "सुरुवातीला निंबोळी तेल ५ मिली/लिटर किंवा ५% निंबोळी अर्काची फवारणी करावी.",
            "chemical": "टेबुकोनॅझोल २५.९% ईसी (२५०-३०० मिली/एकर) किंवा हेक्साकोनॅझोल ५% ईसी (४०० मिली/एकर) २०० लिटर पाण्यात मिसळून फवारावे.",
        },
    },
    "Mosaic": {
        "en": {
            "name": "Soybean Mosaic Virus (SMV)",
            "urgency": "HIGH",
            "symptoms": "Mosaic patterns of light and dark green, puckering, rugosity, and downward curling.",
            "organic": "Rogue out infected plants immediately. Use yellow sticky traps (15-20/acre) to control aphid vectors.",
            "chemical": "No direct cure for virus; control vector aphids with Imidacloprid 17.8% SL @ 50 ml/acre or Thiamethoxam 25% WG @ 40 g/acre.",
        },
        "hi": {
            "name": "सोयाबीन मोज़ेक वायरस (Mosaic)",
            "urgency": "गंभीर (HIGH)",
            "symptoms": "पत्तियों पर हल्के व गहरे हरे रंग के धब्बे, पत्तियां सिकुड़ना और नीचे की ओर मुड़ना।",
            "organic": "संक्रमित पौधों को उखाड़कर नष्ट करें। माहू (एफिड) कीट नियंत्रण के लिए पीले चिपचिपे कार्ड लगाएं।",
            "chemical": "वायरस की सीधी दवा नहीं है; वाहक कीटों के नियंत्रण के लिए इमिडाक्लोप्रिड 17.8% SL (50 मिली/एकड़) का छिड़काव करें।",
        },
        "mr": {
            "name": "सोयाबीन मोझॅक विषाणू (Mosaic)",
            "urgency": "गंभीर (HIGH)",
            "symptoms": "पानांवर फिकट व गडद हिरवे चट्टे दिसतात, पाने आकुंचन पावतात आणि खालच्या दिशेने वळतात.",
            "organic": "रोगट झाडे त्वरित उपटून नष्ट करा. मावा कीटकांच्या नियंत्रणासाठी पिवळे चिकट सापळे लावावेत.",
            "chemical": "विषाणूवर थेट औषध नाही; प्रसारक मावा नियंत्रणासाठी इमिडाक्लोप्रिड १७.८% एसएल (५० मिली/एकर) ची फवारणी करावी.",
        },
    },
    "Caterpillar_Semilooper_Pest": {
        "en": {
            "name": "Semilooper / Defoliator Caterpillar",
            "urgency": "HIGH",
            "symptoms": "Irregular holes chewed in leaves, skeletonized leaf margins, and larvae presence.",
            "organic": "Spray Bacillus thuringiensis (Bt) @ 400 g/acre or Beauveria bassiana @ 1 kg/acre.",
            "chemical": "Spray Chlorantraniliprole 18.5% SC @ 60 ml/acre or Emamectin Benzoate 5% SG @ 80 g/acre.",
        },
        "hi": {
            "name": "सेमीलूपर / पत्ती खाने वाली इल्ली (Caterpillar)",
            "urgency": "गंभीर (HIGH)",
            "symptoms": "पत्तियों में बड़े छेद, कटी हुई पत्तियां और जालीनुमा अवशेष। इल्लियां पत्तियों को तेजी से खाती हैं।",
            "organic": "जैविक नियंत्रण के लिए बीटी (Bt) 400 ग्राम/एकड़ या ब्युवेरिया बासियाना 1 किग्रा/एकड़ का छिड़काव करें।",
            "chemical": "कोराजन (Chlorantraniliprole 18.5% SC) 60 मिली/एकड़ या एमामेक्टिन बेंजोएट 5% SG (80 ग्राम/एकड़) का छिड़काव करें।",
        },
        "mr": {
            "name": "उंटअळी / पाने खाणारी अळी (Caterpillar)",
            "urgency": "गंभीर (HIGH)",
            "symptoms": "पानांना मोठी छिद्रे पडणे, पानांच्या कडा खाणे आणि पानांचा सांगाडा तयार होणे.",
            "organic": "बीटी (Bt) पावडर ४०० ग्रॅम/एकर किंवा बिव्हेरिया बॅसियाना १ किलो/एकर फवारावे.",
            "chemical": "क्लोराँट्रानिलीप्रोल १८.५% एससी (६० मिली/एकर) किंवा इमामेक्टिन बेन्झोएट ५% एसजी (८० ग्रॅम/एकर) ची फवारणी करावी.",
        },
    },
    "Spectoria_Brown_Spot": {
        "en": {
            "name": "Septoria Brown Spot (Septoria glycines)",
            "urgency": "MODERATE",
            "symptoms": "Angular, irregular dark-brown spots with yellow halos, progressing upward from lower leaves.",
            "organic": "Improve field drainage, promote airflow, and spray Trichoderma viride @ 5g/liter.",
            "chemical": "Spray Carbendazim 12% + Mancozeb 63% WP @ 300 g/acre or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 200 ml/acre.",
        },
        "hi": {
            "name": "सेप्टोरिया भूरा पत्ती धब्बा (Brown Spot)",
            "urgency": "मध्यम (MODERATE)",
            "symptoms": "निचली पत्तियों पर भूरे कोणीय धब्बे जिनके चारों ओर पीला घेरा होता है।",
            "organic": "खेत में जल निकासी सुधारें और ट्राइकोडर्मा विरिडी (Trichoderma) 5 ग्राम/लीटर का छिड़काव करें।",
            "chemical": "साफ (Carbendazim 12% + Mancozeb 63% WP) 300 ग्राम/एकड़ का 200 लीटर पानी में छिड़काव करें।",
        },
        "mr": {
            "name": "तपकिरी ठिपके रोग (Brown Spot)",
            "urgency": "मध्यम (MODERATE)",
            "symptoms": "खालच्या पानांवर गडद तपकिरी रंगाचे लहान ठिपके आणि त्यांच्याभोवती पिवळे वलय दिसते.",
            "organic": "शेतातील पाण्याचा निचरा व्यवस्थित ठेवा आणि ट्रायकोडर्मा व्हिरिडी ५ ग्रॅम/लिटर फवारावे.",
            "chemical": "कार्बेन्डाझिम १२% + मॅनकोझेब ६३% डब्ल्यूपी (३०० ग्रॅम/एकर) २०० लिटर पाण्यातून फवारावे.",
        },
    },
    "Frog_Leaf_Eye": {
        "en": {
            "name": "Frogeye Leaf Spot (Cercospora sojina)",
            "urgency": "MODERATE",
            "symptoms": "Circular spots with dark reddish-brown borders and translucent gray/white centers (eye-like appearance).",
            "organic": "Use resistant cultivars and practice balanced potassic nutrition; spray Pseudomonas fluorescens @ 5 ml/liter.",
            "chemical": "Spray Pyraclostrobin 20% WG @ 150-200 g/acre or Propiconazole 25% EC @ 200 ml/acre.",
        },
        "hi": {
            "name": "मेंढक आंख धब्बा रोग (Frogeye Leaf Spot)",
            "urgency": "मध्यम (MODERATE)",
            "symptoms": "गोल से कोणीय धब्बे जिनके किनारे गहरे लाल-भूरे और केंद्र हल्के भूरे/सफेद (आंख जैसे) होते हैं।",
            "organic": "स्यूडोमोनास फ्लोरोसेंस (Pseudomonas fluorescens) 5 मिली/लीटर का छिड़काव करें।",
            "chemical": "प्रोपिकोनाज़ोल 25% EC (200 मिली/एकड़) या पायरैक्लोस्ट्रोबिन 20% WG (150-200 ग्राम/एकड़) का छिड़काव करें।",
        },
        "mr": {
            "name": "बेडूक डोळा ठिपके रोग (Frogeye Leaf Spot)",
            "urgency": "मध्यम (MODERATE)",
            "symptoms": "पानांवर वर्तुळाकार ठिपके ज्यांचे कडा गडद लाल-तपकिरी आणि मध्यभाग पांढरा/राखाडी असतो (डोळ्यासारखा).",
            "organic": "स्यूडोमोनास फ्लोरोसन्स ५ मिली/लिटरची फवारणी करावी.",
            "chemical": "प्रोपिकोनाझोल २५% ईसी (२०० मिली/एकर) किंवा पायरॅक्लोस्ट्रॉबिन २०% डब्ल्यूजी (१५०-२०० ग्रॅम/एकर) फवारावे.",
        },
    },
    "Healthy": {
        "en": {
            "name": "Healthy Soybean Foliage",
            "urgency": "NONE",
            "symptoms": "Clean trifoliate leaf, vigorous green coloration without necrotic lesions or insect damage.",
            "organic": "Maintain preventive nutrition; apply balanced NPK and zinc micronutrient sprays as scheduled.",
            "chemical": "No chemical intervention needed. Monitor weekly during canopy closure.",
        },
        "hi": {
            "name": "स्वस्थ पत्ती (Healthy)",
            "urgency": "कोई रोग नहीं (NONE)",
            "symptoms": "स्वस्थ हरी पत्ती, किसी भी प्रकार के रोग या कीट का कोई लक्षण नहीं है।",
            "organic": "फसल का पोषण संतुलित रखें और सूक्ष्म पोषक तत्वों का समय पर छिड़काव करें।",
            "chemical": "किसी रासायनिक कीटनाशक/फफूंदनाशक की आवश्यकता नहीं है। सप्ताह में एक बार निगरानी रखें।",
        },
        "mr": {
            "name": "निरोगी सोयाबीन पान (Healthy)",
            "urgency": "काहीही नाही (NONE)",
            "symptoms": "स्वच्छ आणि टवटवीत हिरवे पान, कोणताही रोग किंवा कीटक नाही.",
            "organic": "पिकाचे योग्य पोषण ठेवा आणि आवश्यकतेनुसार सूक्ष्म अन्नद्रव्यांची फवारणी करा.",
            "chemical": "कोणत्याही रासायनिक फवारणीची गरज नाही. दर आठवड्याला पिकाची पाहणी करत रहा.",
        },
    },
}


def get_advisory(class_name: str, lang: str = "hi") -> Dict[str, str]:
    """Retrieves advisory for a class in given language (fallback to en)."""
    lang = lang.lower()
    if lang not in ("en", "hi", "mr"):
        lang = "en"
    class_info = ADVISORIES.get(class_name, ADVISORIES["Healthy"])
    return class_info.get(lang, class_info["en"])
