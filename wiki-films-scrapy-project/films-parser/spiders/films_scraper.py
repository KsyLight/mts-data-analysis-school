import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DropItem
import re

class WikiMoviesSpider(scrapy.Spider):
    name = "wiki_movies"
    start_urls = [
        "https://ru.wikipedia.org/wiki/Категория:Фильмы_по_алфавиту"
    ]

    # Расширенные синонимы для нужных полей
    FIELD_SYNONYMS = {
        "genre": ["Жанр", "Жанры", "Тип", "Категория", "Стиль", "Формат", "Субжанр", "Вид"],
        "director": ["Режиссёр", "Режиссер", "Режиссеры", "Режиссёр(ы)", "Режиссура", "Режиссерский состав", "Кинорежиссёр"],
        "country": ["Страна", "Страны", "Происхождение", "География", "Регион", "Место производства", "Производитель"],
        "year": ["Год", "Годы", "Дата выпуска", "Год выпуска", "Год создания", "Дата создания", "Выход"]
    }

    def parse(self, response):
        """
        Извлекает ссылки на страницы с буквами алфавита с начальной страницы категории.
        Берётся первая ссылка из каждого блока.
        """
        alphabet_sections = response.css("ul.ts-module-Индекс_категории-multi-items")
        for section in alphabet_sections:
            first_letter_link = section.css("li:first-child a.external.text::attr(href)").get()
            if first_letter_link:
                yield response.follow(first_letter_link, self.parse_letter_page)

    def parse_letter_page(self, response):
        """
        Парсит страницу для конкретной буквы алфавита, обрабатывая список фильмов и пагинацию.
        """
        for group in response.css("div.mw-category-group"):
            film_links = group.css("ul li a::attr(href)").getall()
            for film_link in film_links:
                yield response.follow(film_link, self.parse_film)
        next_page = response.css("div#mw-pages a:contains('Следующая страница')::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse_letter_page)

    def parse_film(self, response):
        """
        Извлекает данные о фильме со страницы фильма.
        """
        title = response.xpath("//th[contains(@class, 'infobox-above')]/text()").get()
        info_box = response.xpath("//table[contains(@class, 'infobox')]")

        # Извлекаем необходимые поля
        genre_raw = self.extract_field(info_box, self.FIELD_SYNONYMS["genre"])
        # Альтернативный поиск жанра, если не найден в инфобоксе
        if not genre_raw:
            genre_alt = response.xpath("//span[@data-wikidata-property-id='P136']//a/text()").getall()
            if genre_alt:
                genre_raw = ", ".join(genre_alt)

        director_raw = self.extract_field(info_box, self.FIELD_SYNONYMS["director"])
        country_raw = self.extract_field(info_box, self.FIELD_SYNONYMS["country"])
        year_raw = self.extract_field(info_box, self.FIELD_SYNONYMS["year"])

        # Заголовок (Название) — убираем пробелы по краям
        title = title.strip() if title else None

        # Очищаем и нормализуем данные
        genre = self.clean_genre(genre_raw)
        director = self.clean_person_field(director_raw)
        country = self.clean_person_field(country_raw)
        year = self.clean_year(year_raw)

        yield {
            "Название": title,
            "Жанр": genre,
            "Режиссёр": director,
            "Страна": country,
            "Год": year,
            "URL": response.url
        }

    def extract_field(self, info_box, synonyms):
        """
        Ищет в инфобоксе строку, где тег <th> содержит один из синонимов,
        и возвращает текст из соответствующего <td>, исключая <script> и <style>.
        """
        for syn in synonyms:
            row = info_box.xpath(f".//tr[th[contains(text(), '{syn}')]]")
            if row:
                # Извлекаем только текстовые узлы, не входящие в <script> или <style>
                text_list = row.xpath("./td//text()[not(parent::script) and not(parent::style)]").getall()
                text = " ".join(text_list)
                if text.strip():
                    return text.strip()
        return ""

    def clean_text(self, text):
        """
        1. Удаляет CSS-фрагменты вида .mw-parser-output {...}
        2. Удаляет содержимое в квадратных скобках (включая [1], [2], [примечание] и т.п.)
        3. Убирает множественные пробелы.
        """
        if not text:
            return ""

        # Удаляем любые блоки, начинающиеся с .mw-parser-output и продолжающиеся до первой закрывающей фигурной скобки
        text = re.sub(r'\.mw-parser-output.*?\}', '', text, flags=re.DOTALL)

        # Удаляем всё в квадратных скобках (включая [1], [2], [примеч.])
        text = re.sub(r'\[.*?\]', '', text)

        # Убираем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_genre(self, text):
        """
        1. Вызывает общий метод clean_text для удаления CSS и скобок.
        2. Удаляет цифры.
        3. Приводит строку к нижнему регистру.
        """
        text = self.clean_text(text)
        # Удаляем все цифры
        text = re.sub(r'\d+', '', text)
        # Приводим к нижнему регистру
        text = text.lower()
        # Финальная зачистка пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_person_field(self, text):
        """
        Для полей 'Режиссёр' и 'Страна' (или других подобных):
        1. Удаляет CSS и скобки через clean_text.
        2. Возвращает результат без удаления цифр (вдруг нужны).
        """
        return self.clean_text(text)

    def clean_year(self, text):
        """
        Оставляет только 4-значный год (без месяцев и дней).
        Если нет совпадений, возвращает пустую строку.
        """
        text = self.clean_text(text)
        match = re.search(r'\b\d{4}\b', text)
        if match:
            return match.group(0)
        return ""

class FilterIncompleteItemsPipeline:
    # Теперь жанр тоже обязателен
    REQUIRED_FIELDS = ['Название', 'Жанр', 'Режиссёр', 'Страна', 'Год']

    def process_item(self, item, spider):
        for field in self.REQUIRED_FIELDS:
            if not item.get(field) or not item[field].strip():
                raise DropItem(f"Отсутствует обязательное поле {field} в элементе: {item}")
        return item


# Пользовательские настройки
custom_settings = {
    'CLOSESPIDER_TIMEOUT': 600,  # Остановка паука через 10 минут
    'ITEM_PIPELINES': {
        '__main__.FilterIncompleteItemsPipeline': 300,
    },
    'FEED_FORMAT': 'csv',
    'FEED_URI': r'C:\Users\MSI\Documents\GitHub\mts-data-analysis-school\wiki-films-scrapy-project\films_dataset.csv',
    'LOG_LEVEL': 'INFO'
}

if __name__ == '__main__':
    process = CrawlerProcess(settings=custom_settings)
    process.crawl(WikiMoviesSpider)
    process.start()