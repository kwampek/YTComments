# 💬 YouComments

Дообученная модель GPT-2 Large для генерации комментариев и ответов к ним на YouTube.


Основная версия с моделью лежит на hugging face: 
https://huggingface.co/kwampek/YouComments/tree/main


Рекомендую посмотреть `main.ipynb` с подробно описанным ходом обучения. 

### Запуск

0. Перенести `weights.pt` из репозитория выше в папку models
   ```
   $ mv ~/Downloads/weights.pt ./models/weights.pt
   ```
 
1. Скачать `requirements`

   ```
   $ pip install -r requirements.txt
   ```

2. Запустить

   ```
   $ streamlit run streamlit_app.py
   ```
