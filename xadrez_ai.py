import cohere
import requests
import chess
import chess.svg  # Permite renderizar o tabuleiro em SVG
import time
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Buscar as chaves de API do arquivo .env
cohere_api_key = os.getenv("COHERE_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Inicializar o cliente da Cohere
co = cohere.Client(cohere_api_key)

# Função para validar se o movimento é válido no xadrez
def validate_move(board, move):
    try:
        chess_move = board.parse_san(move)
        return True
    except ValueError:
        return False

# Função para fazer uma jogada com a Cohere (geração de texto)
def cohere_move(board):
    board_fen = board.fen()  # Posição do tabuleiro em FEN
    prompt = f"O estado atual do tabuleiro de xadrez em FEN é: {board_fen}. Qual é o seu próximo movimento? Responda no formato SAN (por exemplo, 'e2e4')."

    # Usando a API da Cohere para gerar a próxima jogada
    response = co.generate(
        model='command-xlarge-nightly',  # Modelo da Cohere
        prompt=prompt,
        max_tokens=50,
        temperature=0.7  # Ajusta a criatividade da IA
    )

    # Retorna o movimento sugerido pela IA
    move = response.generations[0].text.strip()

    # Validar o movimento antes de retorná-lo
    if validate_move(board, move):
        return move
    else:
        print(f"Movimento inválido sugerido pela Cohere: {move}")
        return "e2e4"  # Movimento padrão ou pode gerar outro

# Função para fazer uma jogada com a API da Hugging Face
def huggingface_move(board):
    board_fen = board.fen()  # Posição do tabuleiro em FEN
    model = "gpt2"  # Exemplo usando GPT-2 da Hugging Face

    headers = {
        "Authorization": f"Bearer {huggingface_api_key}"
    }

    prompt = (
        f"O estado atual do tabuleiro de xadrez em FEN é: {board_fen}. "
        "Você é um especialista em xadrez. Por favor, forneça o próximo movimento no formato SAN (exemplo: 'e2e4'). "
        "Certifique-se de que o movimento seja válido e não inclua explicações adicionais."
    )

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json={"inputs": prompt},
        timeout=30  # Timeout de 30 segundos
    )

    if response.status_code == 200:
        move = response.json()[0]["generated_text"].strip()

        # Validar o movimento antes de retorná-lo
        if validate_move(board, move):
            return move
        else:
            print(f"Movimento inválido sugerido pela Hugging Face GPT-2: {move}")
            return "e2e4"  # Movimento padrão ou pode gerar outro
    else:
        raise Exception(f"Erro na API Hugging Face: {response.status_code}, {response.text}")

# Função para aplicar o movimento no tabuleiro de xadrez
def make_move(board, move, player_name):
    try:
        chess_move = board.parse_san(move)
        board.push(chess_move)
        print(f"{player_name} fez o movimento: {move}")
    except ValueError:
        print(f"Movimento inválido sugerido por {player_name}: {move}")

# Função para exibir o tabuleiro após cada jogada
def display_board(board):
    print(board)  # Exibe o tabuleiro em texto no terminal
    print("\n")

# Função para salvar o tabuleiro em SVG após cada jogada
def save_svg_board(board, filename="chess_board.svg"):
    with open(filename, "w") as f:
        f.write(chess.svg.board(board=board))
    print(f"Tabuleiro salvo em {filename}")

# Função para simular a partida
def play_game():
    board = chess.Board()

    while not board.is_game_over():
        # Movimento da Cohere
        print("Turno da Cohere:")
        move = cohere_move(board)
        make_move(board, move, "Cohere")
        display_board(board)  # Exibe o tabuleiro no terminal
        save_svg_board(board, "cohere_vs_huggingface.svg")  # Salva o tabuleiro como SVG
        time.sleep(1)  # Pausa para simular tempo de resposta
        if board.is_game_over():
            break

        # Movimento da Hugging Face
        print("Turno da Hugging Face GPT-2:")
        move = huggingface_move(board)
        make_move(board, move, "Hugging Face GPT-2")
        display_board(board)  # Exibe o tabuleiro no terminal
        save_svg_board(board, "cohere_vs_huggingface.svg")  # Salva o tabuleiro como SVG
        time.sleep(1)  # Pausa para simular tempo de resposta
        if board.is_game_over():
            break
    
    print("Jogo terminado!")
    print(f"Resultado final: {board.result()}")

# Inicia o jogo
play_game()
