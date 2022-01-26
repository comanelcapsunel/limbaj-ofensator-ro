# limbaj-ofensator-ro
Scopul proiectului este de a identifica limbajul ofensator în limba română și de a clasifica ”gravitatea” acestuia: warning pentru un limbaj deranjant,
offensive pentru un limbaj deja către limita de a fi considerat obscen. Algoritmul pe care urmează să-l prezentăm pas cu pas utilizează tehnici de
Machine Learning aplicate pe o serie de texte preluate din diverse canale de social media pentru a clasa limbajul folosit de utilizatori. 

# fisiere componente
fisier "badWords" >>> lista cuvinte offensive obtinute cu ajutorul HateBase si prin efort propriu (am luat fiecare litera din alfabet si am fost creativi)
fisier "checkText" >>> corpus adnotat manual pentru ce a fost supus analizei
fisier "checkROText_freq" >>> clasificare empirica + preprocesare text
fisier "checkROText_tfidf" >>> tf-idf + date calitative și cantitative pentru analiza proiectului
fisier "demo task echipa 4" >>> demo audio/video
fisier "documentatie task echipa 4" >>> documentatie amanuntita
fisier "prezentare task4" >>> prezentare PPT

