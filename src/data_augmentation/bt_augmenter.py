from typing import List
from transformers import MarianMTModel, MarianTokenizer
import torch


class BackTranslationAugmenter:
    def __init__(
        self,
        src_lang: str = "ru",
        mid_lang: str = "en",
        rounds: int = 1,
        model_name_template: str = "Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}",
        device: str = None,
        beam_size: int = 5,
    ) -> None:
        """
        src_lang: исходный язык (например, 'ru')
        mid_lang: промежуточный язык (например, 'en')
        rounds: количество проходов back-translation
        beam_size: размер beam search для генерации
        """
        self.src_lang = src_lang
        self.mid_lang = mid_lang
        self.rounds = rounds
        self.beam_size = beam_size

        # Определяем устройство: GPU если доступно
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем модели и токенизаторы
        self._model_src2mid = MarianMTModel.from_pretrained(
            model_name_template.format(src_lang=src_lang, tgt_lang=mid_lang)
        ).to(self.device)
        self._tokenizer_src2mid = MarianTokenizer.from_pretrained(
            model_name_template.format(src_lang=src_lang, tgt_lang=mid_lang)
        )

        self._model_mid2src = MarianMTModel.from_pretrained(
            model_name_template.format(src_lang=mid_lang, tgt_lang=src_lang)
        ).to(self.device)
        self._tokenizer_mid2src = MarianTokenizer.from_pretrained(
            model_name_template.format(src_lang=mid_lang, tgt_lang=src_lang)
        )

    def _translate(self, text: str, model: MarianMTModel, tokenizer: MarianTokenizer) -> str:
        """Выполняет перевод одного текста на модель и токенизатор."""
        batch = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,       # ← обрезание для более 512 токенов
            max_length=512         # ← максимальная длинна
        ).to(self.device)

        translated = model.generate(
            **batch,
            num_beams=self.beam_size,
            max_length=512,
            early_stopping=True,
        )
        output = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return output[0]

    def augment(self, text: str) -> List[str]:
        """
        Возвращает список back-translated вариантов для данного текста.
        Будет выполнено `rounds` проходов:
        text -> mid_lang -> src_lang
        """
        augmented_texts: List[str] = []
        current_texts = [text]

        for _ in range(self.rounds):
            new_texts: List[str] = []
            for t in current_texts:
                try:
                    mid = self._translate(t, self._model_src2mid, self._tokenizer_src2mid)
                    back = self._translate(mid, self._model_mid2src, self._tokenizer_mid2src)
                    new_texts.append(back)
                    augmented_texts.append(back)
                except Exception as e:
                    print(f"[!] Ошибка при back-translation: {e}")
                    continue
            current_texts = new_texts

        return augmented_texts
