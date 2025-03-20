import scrapy

class WikiMoviesSpider(scrapy.Spider):
    name = "wiki_movies"
    start_urls = [
        "https://ru.wikipedia.org/wiki/Категория:Фильмы_по_алфавиту"
    ]

    def parse(self, response):
        """Сначала парсим ссылки на страницы букв алфавита, берём только первую ссылку из каждой группы"""
        
        # Ищем все блоки с буквами (ul.ts-module-Индекс_категории-multi-items)
        alphabet_sections = response.css("ul.ts-module-Индекс_категории-multi-items")

        for section in alphabet_sections:
            # Берём только первую <li> в группе
            first_letter_link = section.css("li:first-child a.external.text::attr(href)").get()
            
            if first_letter_link:
                yield response.follow(first_letter_link, self.parse_letter_page)

    def parse_letter_page(self, response):
        """Парсим все фильмы на странице буквы"""
        # Перебираем все `mw-category-group` (они содержат списки фильмов)
        for group in response.css("div.mw-category-group"):
            for film in group.css("ul li a::attr(href)").getall():
                yield response.follow(film, self.parse_film)

        # Пагинация: если есть "Следующая страница", идём дальше
        next_page = response.css("div#mw-pages a:contains('Следующая страница')::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse_letter_page)

import scrapy

class WikiMoviesSpider(scrapy.Spider):
    name = "wiki_movies"
    start_urls = [
        "https://ru.wikipedia.org/wiki/Категория:Фильмы_по_алфавиту"
    ]

    def parse(self, response):
        """Парсим ссылки на страницы букв алфавита, берём только первую ссылку из каждой группы"""
        
        # Ищем все блоки с буквами (ul.ts-module-Индекс_категории-multi-items)
        alphabet_sections = response.css("ul.ts-module-Индекс_категории-multi-items")

        for section in alphabet_sections:
            # Берём только первую <li> в группе
            first_letter_link = section.css("li:first-child a.external.text::attr(href)").get()
            
            if first_letter_link:
                yield response.follow(first_letter_link, self.parse_letter_page)

    def parse_letter_page(self, response):
        """Парсим все фильмы на странице буквы"""
        # Перебираем все `mw-category-group` (они содержат списки фильмов)
        for group in response.css("div.mw-category-group"):
            for film in group.css("ul li a::attr(href)").getall():
                yield response.follow(film, self.parse_film)

        # Пагинация: если есть "Следующая страница", идём дальше
        next_page = response.css("div#mw-pages a:contains('Следующая страница')::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse_letter_page)

    def parse_film(self, response):
        """Извлекаем данные о фильме"""
        title = response.xpath("//th[contains(@class, 'infobox-above')]/text()").get()

        # Ищем инфобокс (таблица с данными)
        info_box = response.xpath("//table[contains(@class, 'infobox')]")

        # Достаём нужные характеристики
        genre = info_box.xpath(".//tr[th[contains(text(), 'Жанр')]]/td//text()").getall()
        director = info_box.xpath(".//tr[th[contains(text(), 'Режиссёр')]]/td//a/text()").getall()
        country = info_box.xpath(".//tr[th[contains(text(), 'Страна')]]/td//text()").getall()
        year = info_box.xpath(".//tr[th[contains(text(), 'Год')]]/td//text()").get()
        duration = info_box.xpath(".//tr[th[contains(text(), 'Длительность')]]/td//text()").get()
        production = info_box.xpath(".//tr[th[contains(text(), 'Кинокомпания')]]/td//text()").getall()

        yield {
            "Название": title.strip() if title else None,
            "Жанр": ", ".join(genre).strip(),
            "Режиссёр": ", ".join(director).strip(),
            "Страна": ", ".join(country).strip(),
            "Год": year.strip() if year else None,
            "Длительность": duration.strip() if duration else None,
            "Кинокомпания": ", ".join(production).strip(),
        }