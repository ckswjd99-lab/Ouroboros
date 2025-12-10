import os
import csv
import re

def find_mean_jf_files(root_dir):
    """지정된 루트 디렉토리와 그 하위 모든 디렉토리에서 mean_JF.txt 파일을 찾습니다."""
    found_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'mean_JF.txt' in filenames:
            found_files.append(os.path.join(dirpath, 'mean_JF.txt'))
    return found_files

def parse_file_content(file_path):
    """
    파일 내용을 파싱하여 'Dirtiness Map Type: threshold'이고 
    'Dirtiness Threshold: 30'인지 확인하고 관련 데이터를 추출합니다.
    """
    # 파일의 Method 이름과 CSV 헤더 이름을 매핑합니다.
    model_mapping = {
        'evit': 'E-ViT',
        'ours': 'Ours', # 예시: 'ours'라는 메소드가 있다면 'Ours'로 매핑
        'maskvd': 'MaskVD', # 예시
        'stgt': 'STGT' # 예시
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 조건 확인
        if "Dirtiness Map Type: topk" not in content or "Dirtiness Top-K: 1024" not in content or "Model: swin-l" not in content:
            return None

        # 2. 데이터 추출
        parsed_values = {}
        total_flops = 0.0
        model_key = None

        for line in content.splitlines():
            if not line.strip():
                continue
            
            # 정규식을 사용하여 '키: 값' 형식의 데이터를 추출합니다.
            match = re.match(r'([^:]+):\s*(.*)', line)
            if not match:
                continue
            
            key, value = match.groups()
            key = key.strip()
            value = value.strip()

            if key == 'Method':
                model_key = model_mapping.get(value, value) # 매핑된 이름 사용
            elif key == 'Frame Rate':
                parsed_values['FPS'] = int(re.search(r'\d+', value).group())
            elif key == 'Mean Recomp Rate':
                parsed_values['Patch_Keep'] = float(value)
            elif key == 'Mean J':
                parsed_values['J'] = float(value) # Mean J -> J 로 매핑
            elif key == 'Mean F':
                parsed_values['F'] = float(value)  # Mean F -> F 로 매핑
            elif key.endswith('_flops'):
                total_flops += float(value)
        
        if total_flops > 0:
            parsed_values['Computation'] = total_flops

        parsed_values['mJF'] = (parsed_values.get('J', 0) + parsed_values.get('F', 0)) / 2

        # 3. 최종 데이터 구조화
        if model_key and 'FPS' in parsed_values:
            # { 'E-ViT': {'FPS': 6, 'Patch_Keep': 0.3148, ...} } 형식으로 반환
            return {model_key: parsed_values}

    except Exception as e:
        print(f"파일을 읽거나 파싱하는 중 오류 발생 {file_path}: {e}")
    
    return None

def create_csv_from_data(data_list, output_filename):
    """추출된 데이터 리스트를 사용하여 CSV 파일을 생성합니다."""
    if not data_list:
        print("CSV로 작성할 데이터가 없습니다.")
        return

    # 데이터를 FPS 기준으로 그룹화합니다.
    grouped_data = {}
    for data_from_file in data_list:
        for model, values in data_from_file.items():
            fps = values.get('FPS')
            if fps is None:
                continue
            if fps not in grouped_data:
                grouped_data[fps] = {}
            grouped_data[fps][model] = values

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 헤더 작성 (2줄)
        header_row = ['', 'Ours', '', '', '', '', 'E-ViT', '', '', '', '', 'MaskVD', '', '', '', '', 'STGT']
        writer.writerow(header_row)
        sub_header_row = ['FPS', 'Patch_Keep', 'Computation', 'F', 'J', 'mJF', 'Patch_Keep', 'Computation', 'F', 'J', 'mJF', 'Patch_Keep', 'Computation', 'F', 'J', 'mJF', 'Patch_Keep', 'Computation', 'F', 'J', 'mJF']
        writer.writerow(sub_header_row)

        # 데이터 작성
        for fps in sorted(grouped_data.keys()):
            row = [fps]
            # 정의된 모델 순서대로 열을 채웁니다.
            for model in ['Ours', 'E-ViT', 'MaskVD', 'STGT']:
                model_data = grouped_data[fps].get(model, {})
                row.extend([
                    model_data.get('Patch_Keep', ''),
                    model_data.get('Computation', ''),
                    model_data.get('F', ''),
                    model_data.get('J', ''),
                    model_data.get('mJF', '')
                ])
            writer.writerow(row)
            
    print(f"✅ 성공적으로 '{output_filename}' 파일을 생성했습니다.")

if __name__ == "__main__":
    # 검색을 시작할 최상위 디렉토리
    search_directory = 'output/DAVIS2017_trainval'
    # 생성될 CSV 파일 이름
    output_csv_name = 'davis2017_threshold30_performance_summary.csv'

    if not os.path.isdir(search_directory):
        print(f"❌ 오류: '{search_directory}' 디렉토리를 찾을 수 없습니다.")
        print("스크립트를 올바른 위치에서 실행하고 있는지 확인하세요.")
    else:
        # 파일 검색
        mean_jf_files = find_mean_jf_files(search_directory)
        
        if not mean_jf_files:
            print(f"'{search_directory}' 와 그 하위 폴더에서 'mean_JF.txt' 파일을 찾지 못했습니다.")
        else:
            print(f"총 {len(mean_jf_files)}개의 'mean_JF.txt' 파일을 찾았습니다.")
            all_data = []
            for file_path in mean_jf_files:
                # 파일 내용 파싱 및 조건 확인
                parsed_data = parse_file_content(file_path)
                if parsed_data:
                    all_data.append(parsed_data)
            
            # CSV 파일 생성
            create_csv_from_data(all_data, output_csv_name)