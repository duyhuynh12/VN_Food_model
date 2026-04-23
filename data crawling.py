from icrawler.builtin import BingImageCrawler
import os

# Danh sách món ăn bạn chọn
classes = ['Phở bò', 'Bánh mì Việt Nam', 'Cơm tấm', 'Bún bò Huế', 'Gỏi cuốn','Bánh xèo', 'Bún thịt nướng', 'Cao lầu', 'Mì Quảng', 'Bánh bột lọc']

for food in classes:

    save_path = f'data/{food.replace(" ", "_")}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f'--- Đang tải dữ liệu cho món: {food} ---')

    crawler = BingImageCrawler(storage={'root_dir': save_path})
    crawler.crawl(keyword=food, max_num=400)