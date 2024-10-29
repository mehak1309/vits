""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_numbers = '0123456789'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ" + '̃'  # in use
_letters_all = ['Â', 'Ã', 'â', 'ā', 't̺', '³', '×', '´', '½', '(', '←', '☞', 'µ', '￼', ']', 'ˆ', '₦', '&', ':', '∩', '!', '●', '—', '¯', '\\', '¤', '⇒', '‘', '±', '°', '℅', ',', ')', '₤', '№', '£', '٭', '²', '|', '″', '«', '₣', '⅛', '†', '₡', '¢', '=', "'", '»', '¿', '₵', '-', '≡', '₿', '₢', '%', '❀', 'Ⓡ', '٪', '‡', '¸', '‽', '¥', '₠', '₥', '¼', '؛', '•', '/', '…', '₫', '$', '₨', '¶', '۔۔۔', '‰', '¬', '↓', '¹', '█', '¨', '٬', '>', '→', '<', '_', 'ʼ', '“', '₪', '⅞', '©', '¡', '॥', '₺', '₩', '€', '₧', '·', '–', '۔', '*', '⅝', '≤', '~', '₹', 'ª', '[', '”', '`', '#', '↵', 'º', '?', '§', '}', '⅜', '@', '{', '′', '−', '¾', '↑', ';', '+', '^', '"', '？', '،', '٫', '₽', '؟', '।', '.', '¦', '®', '™', 'ऀ', 'ँ', 'ं', 'ः', 'ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ऺ', 'ऻ', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', '्', 'ॎ', 'ॏ', 'ॐ', '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़', 'ॠ', 'ॡ', 'ॢ', 'ॣ', '।', '॥', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९', '॰', 'ॱ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॽ', 'ॾ', 'ॿ', 'ঀ', 'ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'ঌ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', '়', 'ঽ', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ', 'ৗ', 'ড়', 'ঢ়', 'য়', 'ৠ', 'ৡ', 'ৢ', 'ৣ', '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯', 'ৰ', 'ৱ', '৲', '৳', '৴', '৵', '৶', '৷', '৸', '৹', '৺', '৻', 'ৼ', '৽', '৾', 'ਁ', 'ਂ', 'ਃ', 'ਅ', 'ਆ', 'ਇ', 'ਈ', 'ਉ', 'ਊ', 'ਏ', 'ਐ', 'ਓ', 'ਔ', 'ਕ', 'ਖ', 'ਗ', 'ਘ', 'ਙ', 'ਚ', 'ਛ', 'ਜ', 'ਝ', 'ਞ', 'ਟ', 'ਠ', 'ਡ', 'ਢ', 'ਣ', 'ਤ', 'ਥ', 'ਦ', 'ਧ', 'ਨ', 'ਪ', 'ਫ', 'ਬ', 'ਭ', 'ਮ', 'ਯ', 'ਰ', 'ਲ', 'ਲ਼', 'ਵ', 'ਸ਼', 'ਸ', 'ਹ', '਼', 'ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ', '੍', 'ੑ', 'ਖ਼', 'ਗ਼', 'ਜ਼', 'ੜ', 'ਫ਼', '੦', '੧', '੨', '੩', '੪', '੫', '੬', '੭', '੮', '੯', 'ੰ', 'ੱ', 'ੲ', 'ੳ', 'ੴ', 'ੵ', '੶', 'ઁ', 'ં', 'ઃ', 'અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'ઋ', 'ઌ', 'ઍ', 'એ', 'ઐ', 'ઑ', 'ઓ', 'ઔ', 'ક', 'ખ', 'ગ', 'ઘ', 'ઙ', 'ચ', 'છ', 'જ', 'ઝ', 'ઞ', 'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ત', 'થ', 'દ', 'ધ', 'ન', 'પ', 'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 'લ', 'ળ', 'વ', 'શ', 'ષ', 'સ', 'હ', '઼', 'ઽ', 'ા', 'િ', 'ી', 'ુ', 'ૂ', 'ૃ', 'ૄ', 'ૅ', 'ે', 'ૈ', 'ૉ', 'ો', 'ૌ', '્', 'ૐ', 'ૠ', 'ૡ', 'ૢ', 'ૣ', '૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯', '૰', '૱', 'ૹ', 'ૺ', 'ૻ', 'ૼ', '૽', '૾', '૿', 'ଁ', 'ଂ', 'ଃ', 'ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଌ', 'ଏ', 'ଐ', 'ଓ', 'ଔ', 'କ', 'ଖ', 'ଗ', 'ଘ', 'ଙ', 'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ', 'ଟ', 'ଠ', 'ଡ', 'ଢ', 'ଣ', 'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ', 'ପ', 'ଫ', 'ବ', 'ଭ', 'ମ', 'ଯ', 'ର', 'ଲ', 'ଳ', 'ଵ', 'ଶ', 'ଷ', 'ସ', 'ହ', '଼', 'ଽ', 'ା', 'ି', 'ୀ', 'ୁ', 'ୂ', 'ୃ', 'ୄ', 'େ', 'ୈ', 'ୋ', 'ୌ', '୍', '୕', 'ୖ', 'ୗ', 'ଡ଼', 'ଢ଼', 'ୟ', 'ୠ', 'ୡ', 'ୢ', 'ୣ', '୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯', '୰', 'ୱ', '୲', '୳', '୴', '୵', '୶', '୷', 'ஂ', 'ஃ', 'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'க', 'ங', 'ச', 'ஜ', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ன', 'ப', 'ம', 'ய', 'ர', 'ற', 'ல', 'ள', 'ழ', 'வ', 'ஶ', 'ஷ', 'ஸ', 'ஹ', 'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', '்', 'ௐ', 'ௗ', '௦', '௧', '௨', '௩', '௪', '௫', '௬', '௭', '௮', '௯', '௰', '௱', '௲', '௳', '௴', '௵', '௶', '௷', '௸', '௹', '௺', 'ఀ', 'ఁ', 'ం', 'ః', 'ఄ', 'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఌ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'క', 'ఖ', 'గ', 'ఘ', 'ఙ', 'చ', 'ఛ', 'జ', 'ఝ', 'ఞ', 'ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న', 'ప', 'ఫ', 'బ', 'భ', 'మ', 'య', 'ర', 'ఱ', 'ల', 'ళ', 'ఴ', 'వ', 'శ', 'ష', 'స', 'హ', 'ఽ', 'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', '్', 'ౕ', 'ౖ', 'ౘ', 'ౙ', 'ౚ', 'ౠ', 'ౡ', 'ౢ', 'ౣ', '౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯', '౷', '౸', '౹', '౺', '౻', '౼', '౽', '౾', '౿', 'ಀ', 'ಁ', 'ಂ', 'ಃ', '಄', 'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ', '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೠ', 'ೡ', 'ೢ', 'ೣ', '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', 'ೱ', 'ೲ', 'ഀ', 'ഁ', 'ം', 'ഃ', 'ഄ', 'അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'ഌ', 'എ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ', 'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ത', 'ഥ', 'ദ', 'ധ', 'ന', 'ഩ', 'പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ഺ', '഻', '഼', 'ഽ', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'ൄ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ', '്', 'ൎ', '൏', 'ൔ', 'ൕ', 'ൖ', 'ൗ', '൘', '൙', '൚', '൛', '൜', '൝', '൞', 'ൟ', 'ൠ', 'ൡ', 'ൢ', 'ൣ', '൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯', '൰', '൱', '൲', '൳', '൴', '൵', '൶', '൷', '൸', '൹', 'ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ൿ', 'ᰀ', 'ᰁ', 'ᰂ', 'ᰃ', 'ᰄ', 'ᰅ', 'ᰆ', 'ᰇ', 'ᰈ', 'ᰉ', 'ᰊ', 'ᰋ', 'ᰌ', 'ᰍ', 'ᰎ', 'ᰏ', 'ᰐ', 'ᰑ', 'ᰒ', 'ᰓ', 'ᰔ', 'ᰕ', 'ᰖ', 'ᰗ', 'ᰘ', 'ᰙ', 'ᰚ', 'ᰛ', 'ᰜ', 'ᰝ', 'ᰞ', 'ᰟ', 'ᰠ', 'ᰡ', 'ᰢ', 'ᰣ', 'ᰤ', 'ᰥ', 'ᰦ', 'ᰧ', 'ᰨ', 'ᰩ', 'ᰪ', 'ᰫ', 'ᰬ', 'ᰭ', 'ᰮ', 'ᰯ', 'ᰰ', 'ᰱ', 'ᰲ', 'ᰳ', 'ᰴ', 'ᰵ', 'ᰶ', '᰷', '᰻', '᰼', '᰽', '᰾', '᰿', '᱀', '᱁', '᱂', '᱃', '᱄', '᱅', '᱆', '᱇', '᱈', '᱉', 'ᱍ', 'ᱎ', 'ᱏ']
_letters_all = list(sorted(set(_letters_all)))

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_all) + list(_letters_ipa) + list(_numbers)

# Special symbol ids
SPACE_ID = symbols.index(" ")