import os

def save_graph_image(app, filename="state_graph.png"):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "../"))  # 루트 디렉토리로 이동
        image_dir = os.path.join(root_dir, 'image')

        # 'image' 폴더 생성 (존재하지 않으면 생성)
        os.makedirs(image_dir, exist_ok=True)       
        
        # 파일 경로 설정
        file_path = os.path.join(image_dir, filename)
        
        # 그래프 구조 시각화
        graph = app.get_graph()
        
        # Mermaid 다이어그램 PNG 생성
        png_data = graph.draw_mermaid_png()
        
        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(png_data)

        print(f"그래프 이미지가 '{os.path.abspath(file_path)}' 경로에 저장되었습니다")
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")