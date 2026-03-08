from abc import ABC, abstractmethod
from typing import Optional, List
import chess
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F
from typing import Optional, List, Dict
import json

class Player(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        pass


class TransformerPlayer(Player):
    # Regex matches UCI moves like e2e4, e7e8q.
    UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
    # Piece values used only in fallback heuristic (material-based scoring).
    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    # Configurable params
    def __init__(
        self,
        name: str,
        model_id: str = "distilgpt2",
        temperature: float = 0.2,
        max_new_tokens: int = 8,
        retries: int = 3,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retries = retries

        # Lazy-load so initialization remains lightweight for tournament checks.
        # Lazy-load state. Model/tokenizer are not loaded until first move request.
        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

    # Model Loading
    # def _ensure_model(self) -> bool:
    #     if self._model_unavailable:
    #         print("MODEL DOET NIKS")
    #         return False
    #     if self._tokenizer is not None and self._model is not None:
    #         # print(self._tokenizer)
    #         # print(self._model)
    #         return True

    #     try:
    #         self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    #         if self._tokenizer.pad_token is None:
    #             self._tokenizer.pad_token = self._tokenizer.eos_token

    #         self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
    #         self._model.eval()
    #         if torch.cuda.is_available():
    #             self._model = self._model.to("cuda")
    #         return True
    #     except Exception:
    #         self._model_unavailable = True
    #         self._tokenizer = None
    #         self._model = None
    #         return False

        def _ensure_model(self) -> bool:
            if self._model_unavailable:
                return False
            if self._tokenizer is not None and self._model is not None:
                return True

            try:
                adapter_path = "distilgpt2-chess-adapter"  # local folder (or HF repo id)
                base_model = "distilgpt2"

                self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                base = AutoModelForCausalLM.from_pretrained(base_model)
                self._model = PeftModel.from_pretrained(base, adapter_path)

                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                return True
            except Exception as e:
                print("MODEL LOAD FAILED:", repr(e))
                self._model_unavailable = True
                self._tokenizer = None
                self._model = None
                return False
    # Prompt Building
    def _build_prompt(self, fen: str, legal_uci: List[str]) -> str:
        legal_hint = ", ".join(legal_uci[:40])
        # return(
        #         "You are a chess engine.\n"
        #         "Return exactly one legal move in UCI format.\n"
        #         "No explanations, no punctuation, no extra text.\n\n"
        #         f"FEN: {fen}\n"
        #         f"Legal moves include: {legal_hint}\n"
        #         "Move:"
        #     )
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Legal moves include: {legal_hint}

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""
    
            

    # Text-to-Move Extraction
    def _extract_uci(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = self.UCI_RE.search(text) # Regex-search first UCI-like substring.
        if not match:
            return None
        return match.group(1).lower() # Return lowercase move if found, else None.
    
################### Fallback Helpers ###################
    #  Returns captured piece value for a move (special-case en passant).
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        # En passant has no piece on destination square.
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0
        return self.PIECE_VALUE[captured.piece_type]
    
    # Heuristic move chooser:
    def _fallback_best_legal(self, board: chess.Board, legal_moves: List[chess.Move]) -> str:
        best_score = -10**9 # Start with very low best score.
        best_move = legal_moves[0]
        # For each legal move, compute score.
        for move in legal_moves:
            score = 0
            mover = board.piece_at(move.from_square)
            mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0
            
            # Reward captures (10 * captured - mover_value).
            if board.is_capture(move):
                captured_value = self._captured_piece_value(board, move)
                score += 10 * captured_value - mover_value
            
            # Reward promotions.
            if move.promotion:
                score += self.PIECE_VALUE.get(move.promotion, 0) * 10

            # Simulate move (push), reward checkmate massively, check moderately, then pop.
            board.push(move)
            if board.is_checkmate():
                score += 100000
            elif board.is_check():
                score += 50
            board.pop()

            # Mild centralization preference. Small reward for central squares.
            file_idx = chess.square_file(move.to_square)
            rank_idx = chess.square_rank(move.to_square)
            score += 3 - abs(3.5 - file_idx)
            score += 3 - abs(3.5 - rank_idx)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci() # Return best move UCI.

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen) # Build board from FEN.
        legal_moves = list(board.legal_moves) # Get legal moves
        if not legal_moves: 
            return None # If no legal moves, return None
        legal_uci = [mv.uci() for mv in legal_moves] # Build legal UCI list/set
        legal_set = set(legal_uci) 
        print(legal_set)
        if self._ensure_model(): # If model loads
            # print(self._ensure_model())
            # print("MODEL LOADS")
            # print("retries =", self.retries, flush=True)
            # print("build_prompt args check", flush=True)
            prompt = self._build_prompt(fen, legal_uci)  # important
            # print("prompt built", flush=True)
            # prompt = self._build_prompt(fen) # Build prompt
            for _ in range(self.retries): # try generations up to retries
                try:
                    # print("I AM TRYING")
                    inputs = self._tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True,
                            temperature=self.temperature,
                            pad_token_id=self._tokenizer.pad_token_id,
                        )

                    decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True) # decode output
                    # print(f"[RAW MODEL OUTPUT] {decoded!r}")
                    if decoded.startswith(prompt): # Strip prompt prefix.
                        decoded = decoded[len(prompt):]
                    # print(f"[MODEL AFTER PROMPT STRIP] {decoded!r}")
                    candidate = self._extract_uci(decoded) # Extract candidate UCI by regex.
                    # print(f"[EXTRACTED CANDIDATE] {candidate!r}")
                    # print(f"MODELS MOVE TRY {_} = {candidate}")
                    if candidate in legal_set: # If candidate is legal, return it.
                        # print(f"THE MOVE I CREATED IS FUCKING LEGAL BITCH -> {candidate}")
                        return candidate 
                except Exception:
                    break

        # Never return an illegal move: strong fallback policy.
        return self._fallback_best_legal(board, legal_moves)


class TransformerPlayer_2(Player):
    # Regex matches UCI moves like e2e4, e7e8q.
    UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
    # Piece values used only in fallback heuristic (material-based scoring).
    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    # Configurable params
    def __init__(
        self,
        name: str,
        model_id: str = "distilgpt2",
        adapter_path: str = "distilgpt2-chess-lora-adapter",
    ):
        super().__init__(name)
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

    # Model Loading
    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._tokenizer is not None and self._model is not None:
            return True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model = PeftModel.from_pretrained(base, self.adapter_path)

            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            return True
        except Exception as e:
            print("MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            return False
            
    def score_move(self, prompt: str, move: str) -> float:
        device = self._model.device

        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = self._tokenizer(prompt + move, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = self._model(full_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, prompt_len:]
        pred_log_probs = log_probs[:, prompt_len - 1:-1, :]

        token_logps = torch.gather(
            pred_log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_logps.sum().item()
        
    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = [mv.uci() for mv in board.legal_moves]

        # Match training format
        prompt = f"FEN: {fen}\nEval: +0.000\nMove: "

        best_move = legal_moves[0]
        best_score = float("-inf")

        for mv in legal_moves:
            score = self.score_move(prompt, mv)
            # print(f"move: {mv} | score: {score}")
            if score > best_score:
                best_score = score
                best_move = mv

        return best_move
    
    # Text-to-Move Extraction
    def _extract_uci(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = self.UCI_RE.search(text) # Regex-search first UCI-like substring.
        if not match:
            return None
        return match.group(1).lower() # Return lowercase move if found, else None.
    
################### Fallback Helpers ###################
    #  Returns captured piece value for a move (special-case en passant).
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        # En passant has no piece on destination square.
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0
        return self.PIECE_VALUE[captured.piece_type]
    
    # Heuristic move chooser:
    def _fallback_best_legal(self, board: chess.Board, legal_moves: List[chess.Move]) -> str:
        best_score = -10**9 # Start with very low best score.
        best_move = legal_moves[0]
        # For each legal move, compute score.
        for move in legal_moves:
            score = 0
            mover = board.piece_at(move.from_square)
            mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0
            
            # Reward captures (10 * captured - mover_value).
            if board.is_capture(move):
                captured_value = self._captured_piece_value(board, move)
                score += 10 * captured_value - mover_value
            
            # Reward promotions.
            if move.promotion:
                score += self.PIECE_VALUE.get(move.promotion, 0) * 10

            # Simulate move (push), reward checkmate massively, check moderately, then pop.
            board.push(move)
            if board.is_checkmate():
                score += 100000
            elif board.is_check():
                score += 50
            board.pop()

            # Mild centralization preference. Small reward for central squares.
            file_idx = chess.square_file(move.to_square)
            rank_idx = chess.square_rank(move.to_square)
            score += 3 - abs(3.5 - file_idx)
            score += 3 - abs(3.5 - rank_idx)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci() # Return best move UCI.

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)
    

class TransformerPlayer_3(Player):
    # Regex matches UCI moves like e2e4, e7e8q.
    UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
    # Piece values used only in fallback heuristic (material-based scoring).
    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    # Configurable params
    def __init__(
        self,
        name: str,
        model_id: str = "distilgpt2",
        adapter_path: str = "distilgpt2-chess-lora-adapter",
        alpha: float = 30.0,
        beta: float = 1.0,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.adapter_path = adapter_path

        # hybrid scoring weights
        self.alpha = alpha
        self.beta = beta

        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

    # Model Loading
    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._tokenizer is not None and self._model is not None:
            return True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model = PeftModel.from_pretrained(base, self.adapter_path)

            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            return True
        except Exception as e:
            print("MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            return False
            
    def score_move(self, prompt: str, move: str) -> float:
        device = self._model.device

        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = self._tokenizer(prompt + move, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = self._model(full_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, prompt_len:]
        pred_log_probs = log_probs[:, prompt_len - 1:-1, :]

        token_logps = torch.gather(
            pred_log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_logps.sum().item()
    
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        # En passant has no piece on destination square
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0

        return self.PIECE_VALUE[captured.piece_type]

    def heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0
        mover = board.piece_at(move.from_square)
        mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

        if board.is_capture(move):
            captured_value = self._captured_piece_value(board, move)
            score += 10 * captured_value - mover_value

        if move.promotion:
            score += self.PIECE_VALUE.get(move.promotion, 0) * 10

        board.push(move)
        if board.is_checkmate():
            score += 100000
        elif board.is_check():
            score += 50
        board.pop()

        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        score += 3 - abs(3.5 - file_idx)
        score += 3 - abs(3.5 - rank_idx)
        return score

    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        prompt = f"FEN: {fen}\nEval: +0.000\nMove: "

        model_scores = []
        heuristic_scores = []

        for mv in legal_moves:
            uci = mv.uci()

            # transformer score
            try:
                mscore = self.score_move(prompt, " " + uci)
            except Exception:
                mscore = -1e9

            # heuristic score
            hscore = self.heuristic_score(board, mv)

            model_scores.append(mscore)
            heuristic_scores.append(hscore)

        # normalize model scores so they are easier to combine
        m_min = min(model_scores)
        m_max = max(model_scores)
        if m_max > m_min:
            norm_model_scores = [(s - m_min) / (m_max - m_min) for s in model_scores]
        else:
            norm_model_scores = [0.0 for _ in model_scores]

        # combine
        best_idx = 0
        best_score = float("-inf")

        # alpha = 4.0 # weight for transformer
        # beta = 1.0     # weight for heuristic

        for i, mv in enumerate(legal_moves):
            final_score = self.alpha * norm_model_scores[i] + self.beta * heuristic_scores[i]

            # print(
            #     f"move: {mv.uci()} | "
            #     f"model: {model_scores[i]:.3f} | "
            #     f"norm_model: {norm_model_scores[i]:.3f} | "
            #     f"heuristic: {heuristic_scores[i]:.3f} | "
            #     f"final: {final_score:.3f}"
            # )

            if final_score > best_score:
                best_score = final_score
                best_idx = i
        # print("\n") 
        return legal_moves[best_idx].uci()
    
    # Text-to-Move Extraction
    def _extract_uci(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = self.UCI_RE.search(text) # Regex-search first UCI-like substring.
        if not match:
            return None
        return match.group(1).lower() # Return lowercase move if found, else None.
    

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)    
    

class TransformerPlayer_4(Player):
    # Regex matches UCI moves like e2e4, e7e8q.
    UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
    # Piece values used only in fallback heuristic (material-based scoring).
    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    # Configurable params
    def __init__(
        self,
        name: str,
        model_id: str = "gpt2",
        adapter_path: str = "distilgpt2-chess-lora-adapter_all",
        alpha: float = 30.0,
        beta: float = 1.0,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.adapter_path = adapter_path

        # hybrid scoring weights
        self.alpha = alpha
        self.beta = beta

        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

    # Model Loading
    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._tokenizer is not None and self._model is not None:
            return True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model = PeftModel.from_pretrained(base, self.adapter_path)

            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            return True
        except Exception as e:
            print("MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            return False
            
    def score_move(self, prompt: str, move: str) -> float:
        device = self._model.device

        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = self._tokenizer(prompt + move, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = self._model(full_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, prompt_len:]
        pred_log_probs = log_probs[:, prompt_len - 1:-1, :]

        token_logps = torch.gather(
            pred_log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_logps.sum().item()
    
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        # En passant has no piece on destination square
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0

        return self.PIECE_VALUE[captured.piece_type]

    def heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0
        mover = board.piece_at(move.from_square)
        mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

        if board.is_capture(move):
            captured_value = self._captured_piece_value(board, move)
            score += 10 * captured_value - mover_value

        if move.promotion:
            score += self.PIECE_VALUE.get(move.promotion, 0) * 10

        board.push(move)
        if board.is_checkmate():
            score += 100000
        elif board.is_check():
            score += 50
        board.pop()

        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        score += 3 - abs(3.5 - file_idx)
        score += 3 - abs(3.5 - rank_idx)
        return score

    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        prompt = f"FEN: {fen}\nMove: "

        model_scores = []
        heuristic_scores = []

        for mv in legal_moves:
            uci = mv.uci()

            # transformer score
            try:
                mscore = self.score_move(prompt, " " + uci)
            except Exception:
                mscore = -1e9

            # heuristic score
            hscore = self.heuristic_score(board, mv)

            model_scores.append(mscore)
            heuristic_scores.append(hscore)

        # normalize model scores so they are easier to combine
        m_min = min(model_scores)
        m_max = max(model_scores)
        if m_max > m_min:
            norm_model_scores = [(s - m_min) / (m_max - m_min) for s in model_scores]
        else:
            norm_model_scores = [0.0 for _ in model_scores]

        # combine
        best_idx = 0
        best_score = float("-inf")

        # alpha = 4.0 # weight for transformer
        # beta = 1.0     # weight for heuristic

        for i, mv in enumerate(legal_moves):
            final_score = self.alpha * norm_model_scores[i] + self.beta * heuristic_scores[i]

            # print(
            #     f"move: {mv.uci()} | "
            #     f"model: {model_scores[i]:.3f} | "
            #     f"norm_model: {norm_model_scores[i]:.3f} | "
            #     f"heuristic: {heuristic_scores[i]:.3f} | "
            #     f"final: {final_score:.3f}"
            # )
           
            if final_score > best_score:
                best_score = final_score
                best_idx = i
        # print("\n") 
        return legal_moves[best_idx].uci()
    
    # Text-to-Move Extraction
    def _extract_uci(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = self.UCI_RE.search(text) # Regex-search first UCI-like substring.
        if not match:
            return None
        return match.group(1).lower() # Return lowercase move if found, else None.
    

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)      

class TransformerPlayer_5(Player):
    # Regex matches UCI moves like e2e4, e7e8q.
    UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
    # Piece values used only in fallback heuristic (material-based scoring).
    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    # Configurable params
    def __init__(
        self,
        name: str,
        model_id: str = "distilgpt2",
        adapter_path: str = "distilgpt2-chess-lora-adapter_all",
        alpha: float = 30.0,
        beta: float = 1.0,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.adapter_path = adapter_path

        # hybrid scoring weights
        self.alpha = alpha
        self.beta = beta

        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

    # Model Loading
    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._tokenizer is not None and self._model is not None:
            return True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model = PeftModel.from_pretrained(base, self.adapter_path)

            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            return True
        except Exception as e:
            print("MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            return False
            
    def score_move(self, prompt: str, move: str) -> float:
        device = self._model.device

        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = self._tokenizer(prompt + move, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = self._model(full_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, prompt_len:]
        pred_log_probs = log_probs[:, prompt_len - 1:-1, :]

        token_logps = torch.gather(
            pred_log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_logps.mean().item()
    
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        # En passant has no piece on destination square
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0

        return self.PIECE_VALUE[captured.piece_type]

    def heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0
        mover = board.piece_at(move.from_square)
        mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

        if board.is_capture(move):
            captured_value = self._captured_piece_value(board, move)
            score += 10 * captured_value - mover_value

        if move.promotion:
            score += self.PIECE_VALUE.get(move.promotion, 0) * 10

        board.push(move)
        if board.is_checkmate():
            score += 100000
        elif board.is_check():
            score += 50
        board.pop()

        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        score += 3 - abs(3.5 - file_idx)
        score += 3 - abs(3.5 - rank_idx)
        return score

    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        prompt = f"FEN: {fen}\nMove: "

        model_scores = []
        heuristic_scores = []

        for mv in legal_moves:
            uci = mv.uci()

            # transformer score
            try:
                mscore = self.score_move(prompt, " " + uci)
            except Exception:
                mscore = -1e9

            # heuristic score
            hscore = self.heuristic_score(board, mv)

            model_scores.append(mscore)
            heuristic_scores.append(hscore)

        # normalize model scores so they are easier to combine
        m_min = min(model_scores)
        m_max = max(model_scores)
        if m_max > m_min:
            norm_model_scores = [(s - m_min) / (m_max - m_min) for s in model_scores]
        else:
            norm_model_scores = [0.0 for _ in model_scores]

        # combine
        best_idx = 0
        best_score = float("-inf")

        # alpha = 4.0 # weight for transformer
        # beta = 1.0     # weight for heuristic
        h_min = min(heuristic_scores)
        h_max = max(heuristic_scores)

        if h_max > h_min:
            norm_heuristic = [(s - h_min) / (h_max - h_min) for s in heuristic_scores]
        else:
            norm_heuristic = [0.0 for _ in heuristic_scores]

        for i, mv in enumerate(legal_moves):
            final_score = self.alpha * norm_model_scores[i] + self.beta * norm_heuristic[i]

            # print(
            #     f"move: {mv.uci()} | "
            #     f"model: {model_scores[i]:.3f} | "
            #     f"norm_model: {norm_model_scores[i]:.3f} | "
            #     f"heuristic: {norm_heuristic[i]:.3f} | "
            #     f"final: {final_score:.3f}"
            # )
           
            if final_score > best_score:
                best_score = final_score
                best_idx = i
        # print("\n") 
        return legal_moves[best_idx].uci()
    
    # Text-to-Move Extraction
    def _extract_uci(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = self.UCI_RE.search(text) # Regex-search first UCI-like substring.
        if not match:
            return None
        return match.group(1).lower() # Return lowercase move if found, else None.
    

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)  

class EncoderTransformerPlayer(Player):
    UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)

    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(
        self,
        name: str,
        model_id: str = "bert-chess-model",
        move_vocab_path: str = "move_to_id.json",
        use_board_grid: bool = False,
    ):
        super().__init__(name)
        self.model_path = model_id
        self.move_vocab_path = move_vocab_path
        self.use_board_grid = use_board_grid

        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

        self.move_to_id: Dict[str, int] = {}
        self.id_to_move: Dict[int, str] = {}

    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._tokenizer is not None and self._model is not None and self.move_to_id:
            return True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            with open(self.move_vocab_path, "r") as f:
                self.move_to_id = json.load(f)

            self.id_to_move = {idx: move for move, idx in self.move_to_id.items()}
            return True

        except Exception as e:
            print("ENCODER MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            self.move_to_id = {}
            self.id_to_move = {}
            return False

    def _fen_to_8x8_text(self, fen: str) -> str:
        board = chess.Board(fen)
        rows = []

        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                row.append(piece.symbol() if piece else ".")
            rows.append(" ".join(row))

        side = "white" if board.turn == chess.WHITE else "black"
        castling = board.castling_xfen() if board.castling_rights else "-"
        ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"

        return (
            f"Side to move: {side}\n"
            f"Castling: {castling}\n"
            f"En passant: {ep}\n"
            f"Board:\n" + "\n".join(rows)
        )

    def _build_input(self, fen: str) -> str:
        if self.use_board_grid:
            return self._fen_to_8x8_text(fen)
        return f"FEN: {fen}"

    def _score_legal_moves(self, fen: str, legal_uci: List[str]) -> Dict[str, float]:
        device = self._model.device
        text = self._build_input(fen)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]

        legal_scores = {}
        for mv in legal_uci:
            idx = self.move_to_id.get(mv)
            if idx is not None and idx < logits.shape[0]:
                legal_scores[mv] = logits[idx].item()

        return legal_scores

    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0
        return self.PIECE_VALUE[captured.piece_type]

    def _fallback_best_legal(self, board: chess.Board, legal_moves: List[chess.Move]) -> str:
        best_score = -10**9
        best_move = legal_moves[0]

        for move in legal_moves:
            score = 0
            mover = board.piece_at(move.from_square)
            mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

            if board.is_capture(move):
                captured_value = self._captured_piece_value(board, move)
                score += 10 * captured_value - mover_value

            if move.promotion:
                score += self.PIECE_VALUE.get(move.promotion, 0) * 10

            board.push(move)
            if board.is_checkmate():
                score += 100000
            elif board.is_check():
                score += 50
            board.pop()

            file_idx = chess.square_file(move.to_square)
            rank_idx = chess.square_rank(move.to_square)
            score += 3 - abs(3.5 - file_idx)
            score += 3 - abs(3.5 - rank_idx)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        legal_uci = [mv.uci() for mv in legal_moves]

        if self._ensure_model():
            try:
                legal_scores = self._score_legal_moves(fen, legal_uci)
                if legal_scores:
                    return max(legal_scores.items(), key=lambda x: x[1])[0]
            except Exception as e:
                print("ENCODER SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)
    

class EncoderTransformerPlayer_2(Player):
    UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)

    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(
        self,
        name: str,
        model_id: str = "bert-chess-model",
        move_vocab_path: str = "move_to_id.json",
        use_board_grid: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__(name)
        self.model_path = model_id
        self.move_vocab_path = move_vocab_path
        self.use_board_grid = use_board_grid
        self.alpha = alpha
        self.beta = beta

        self._tokenizer = None
        self._model = None
        self._model_unavailable = False

        self.move_to_id: Dict[str, int] = {}
        self.id_to_move: Dict[int, str] = {}

    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._tokenizer is not None and self._model is not None and self.move_to_id:
            return True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            with open(self.move_vocab_path, "r") as f:
                self.move_to_id = json.load(f)

            self.id_to_move = {idx: move for move, idx in self.move_to_id.items()}
            return True

        except Exception as e:
            print("ENCODER MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            self.move_to_id = {}
            self.id_to_move = {}
            return False

    def _fen_to_8x8_text(self, fen: str) -> str:
        board = chess.Board(fen)
        rows = []

        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                row.append(piece.symbol() if piece else ".")
            rows.append(" ".join(row))

        side = "white" if board.turn == chess.WHITE else "black"
        castling = board.castling_xfen() if board.castling_rights else "-"
        ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"

        return (
            f"Side to move: {side}\n"
            f"Castling: {castling}\n"
            f"En passant: {ep}\n"
            f"Board:\n" + "\n".join(rows)
        )

    def _build_input(self, fen: str) -> str:
        if self.use_board_grid:
            return self._fen_to_8x8_text(fen)
        return f"FEN: {fen}"

    def _score_legal_moves(self, fen: str, legal_uci: List[str]) -> Dict[str, float]:
        """
        Returns raw model logits for legal moves only.
        """
        device = self._model.device
        text = self._build_input(fen)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]

        legal_scores = {}
        for mv in legal_uci:
            idx = self.move_to_id.get(mv)
            if idx is not None and idx < logits.shape[0]:
                legal_scores[mv] = logits[idx].item()

        return legal_scores

    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0
        return self.PIECE_VALUE[captured.piece_type]

    def heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0
        mover = board.piece_at(move.from_square)
        mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

        if board.is_capture(move):
            captured_value = self._captured_piece_value(board, move)
            score += 10 * captured_value - mover_value

        if move.promotion:
            score += self.PIECE_VALUE.get(move.promotion, 0) * 10

        board.push(move)
        if board.is_checkmate():
            score += 100000
        # elif board.is_check():
        #     score += 10
        board.pop()

        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        score += 3 - abs(3.5 - file_idx)
        score += 3 - abs(3.5 - rank_idx)

        return score

    def _fallback_best_legal(self, board: chess.Board, legal_moves: List[chess.Move]) -> str:
        best_score = -10**9
        best_move = legal_moves[0]

        for move in legal_moves:
            score = self.heuristic_score(board, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()

    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        legal_uci = [mv.uci() for mv in legal_moves]

        # model scores for legal moves
        legal_scores = self._score_legal_moves(fen, legal_uci)

        # if model cannot score anything, fallback
        if not legal_scores:
            return self._fallback_best_legal(board, legal_moves)

        model_scores = []
        heuristic_scores = []

        filtered_moves = []
        for mv in legal_moves:
            uci = mv.uci()
            if uci not in legal_scores:
                continue

            filtered_moves.append(mv)
            model_scores.append(legal_scores[uci])
            heuristic_scores.append(self.heuristic_score(board, mv))

        if not filtered_moves:
            return self._fallback_best_legal(board, legal_moves)

        # normalize model scores
        m_min = min(model_scores)
        m_max = max(model_scores)
        if m_max > m_min:
            norm_model_scores = [(s - m_min) / (m_max - m_min) for s in model_scores]
        else:
            norm_model_scores = [0.0 for _ in model_scores]

        # normalize heuristic scores
        h_min = min(heuristic_scores)
        h_max = max(heuristic_scores)
        if h_max > h_min:
            norm_heuristic_scores = [(s - h_min) / (h_max - h_min) for s in heuristic_scores]
        else:
            norm_heuristic_scores = [0.0 for _ in heuristic_scores]

        best_idx = 0
        best_score = float("-inf")

        for i, mv in enumerate(filtered_moves):
            final_score = (
                self.alpha * norm_model_scores[i]
                + self.beta * norm_heuristic_scores[i]
            )

            # Uncomment for debugging
            # print(
            #     f"{mv.uci():6s} | "
            #     f"model={model_scores[i]:8.4f} | "
            #     f"norm_model={norm_model_scores[i]:6.3f} | "
            #     f"heuristic={heuristic_scores[i]:8.3f} | "
            #     f"norm_heur={norm_heuristic_scores[i]:6.3f} | "
            #     f"final={final_score:8.4f}"
            # )

            if final_score > best_score:
                best_score = final_score
                best_idx = i

        return filtered_moves[best_idx].uci()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("ENCODER SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)

import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(
        self,
        num_moves,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        self.square_embed = nn.Embedding(13, d_model)     # 0..12 piece IDs
        self.pos_embed = nn.Embedding(64, d_model)
        self.side_embed = nn.Embedding(2, d_model)
        self.castling_embed = nn.Embedding(2, d_model)
        self.ep_embed = nn.Embedding(65, d_model)         # 0..63 or 64=no ep

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # extra state tokens: side, 4 castling, ep
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Linear(d_model, num_moves)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),   # output in [-1,1]
        )

    def forward(self, squares, side_to_move, castling, ep_square):
        B = squares.size(0)
        device = squares.device

        pos_ids = torch.arange(64, device=device).unsqueeze(0).expand(B, 64)

        x = self.square_embed(squares) + self.pos_embed(pos_ids)

        side_tok = self.side_embed(side_to_move).unsqueeze(1)              # [B,1,D]
        castling_tok = self.castling_embed(castling)                       # [B,4,D]
        ep_tok = self.ep_embed(ep_square).unsqueeze(1)                     # [B,1,D]
        cls_tok = self.cls_token.expand(B, 1, -1)                          # [B,1,D]

        x = torch.cat([cls_tok, side_tok, castling_tok, ep_tok, x], dim=1) # [B,71,D]
        x = self.encoder(x)
        x = self.norm(x)

        pooled = x[:, 0]   # CLS token

        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)

        return policy_logits, value


class ChessTransformerPlayer(Player):
    """
    Player for a plain PyTorch ChessTransformer model saved with:
        torch.save(model.state_dict(), "chess_transformer.pt")

    IMPORTANT:
    - This assumes the ChessTransformer class is already defined exactly
      as it was during training.
    - This also assumes move_to_id.json is the SAME vocabulary used in training.
    """

    UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)

    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    PIECE_TO_ID = {
        None: 0,
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
        "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
    }

    def __init__(
        self,
        name: str,
        model_path: str = "chess_transformer.pt",
        move_vocab_path: str = "move_to_id.json",
        alpha: float = 1.0,
        beta: float = 1.0,
        debug: bool = False,
    ):
        super().__init__(name)
        self.model_path = model_path
        self.move_vocab_path = move_vocab_path
        self.alpha = alpha
        self.beta = beta
        self.debug = debug

        self._model = None
        self._model_unavailable = False
        self.move_to_id: Dict[str, int] = {}
        self.id_to_move: Dict[int, str] = {}

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    # ---------------------------
    # Model loading
    # ---------------------------
    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._model is not None and self.move_to_id:
            return True

        try:
            with open(self.move_vocab_path, "r") as f:
                self.move_to_id = json.load(f)

            self.id_to_move = {idx: move for move, idx in self.move_to_id.items()}

            # IMPORTANT: ChessTransformer must already be defined in your notebook/script.
            self._model = ChessTransformer(num_moves=len(self.move_to_id))
            self._model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self._model.to(self.device)
            self._model.eval()
            return True

        except Exception as e:
            print("MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._model = None
            self.move_to_id = {}
            self.id_to_move = {}
            return False

    # ---------------------------
    # FEN -> tensor features
    # ---------------------------
    def _fen_to_features(self, fen: str):
        board = chess.Board(fen)

        squares = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            squares.append(self.PIECE_TO_ID[piece.symbol()] if piece else 0)

        side_to_move = 1 if board.turn == chess.WHITE else 0

        castling = [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ]

        ep_square = board.ep_square if board.ep_square is not None else 64

        return {
            "squares": torch.tensor(squares, dtype=torch.long),
            "side_to_move": torch.tensor(side_to_move, dtype=torch.long),
            "castling": torch.tensor(castling, dtype=torch.long),
            "ep_square": torch.tensor(ep_square, dtype=torch.long),
        }

    # ---------------------------
    # Model scoring
    # ---------------------------
    def _score_legal_moves(self, fen: str, legal_uci: List[str]) -> Dict[str, float]:
        """
        Returns raw policy logits for legal moves only.
        """
        feats = self._fen_to_features(fen)

        batch = {
            "squares": feats["squares"].unsqueeze(0).to(self.device),
            "side_to_move": feats["side_to_move"].unsqueeze(0).to(self.device),
            "castling": feats["castling"].unsqueeze(0).to(self.device),
            "ep_square": feats["ep_square"].unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            policy_logits, value_pred = self._model(
                batch["squares"],
                batch["side_to_move"],
                batch["castling"],
                batch["ep_square"],
            )

        logits = policy_logits[0]  # shape: [num_moves]

        legal_scores = {}
        for mv in legal_uci:
            idx = self.move_to_id.get(mv)
            if idx is not None and idx < logits.shape[0]:
                legal_scores[mv] = logits[idx].item()

        return legal_scores

    # ---------------------------
    # Heuristic scoring
    # ---------------------------
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0
        return self.PIECE_VALUE[captured.piece_type]

    def heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        mover = board.piece_at(move.from_square)
        mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

        if board.is_capture(move):
            captured_value = self._captured_piece_value(board, move)
            score += 10 * captured_value - mover_value

        if move.promotion:
            score += self.PIECE_VALUE.get(move.promotion, 0) * 10

        board.push(move)
        if board.is_checkmate():
            score += 100000
        elif board.is_check():
            score += 50
        board.pop()

        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        score += 3 - abs(3.5 - file_idx)
        score += 3 - abs(3.5 - rank_idx)

        return score

    def _fallback_best_legal(self, board: chess.Board, legal_moves: List[chess.Move]) -> str:
        best_score = -10**9
        best_move = legal_moves[0]

        for move in legal_moves:
            score = self.heuristic_score(board, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()

    # ---------------------------
    # Hybrid selection
    # ---------------------------
    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        legal_uci = [mv.uci() for mv in legal_moves]

        legal_scores = self._score_legal_moves(fen, legal_uci)

        if not legal_scores:
            return self._fallback_best_legal(board, legal_moves)

        filtered_moves = []
        model_scores = []
        heuristic_scores = []

        for mv in legal_moves:
            uci = mv.uci()
            if uci not in legal_scores:
                continue

            filtered_moves.append(mv)
            model_scores.append(legal_scores[uci])
            heuristic_scores.append(self.heuristic_score(board, mv))

        if not filtered_moves:
            return self._fallback_best_legal(board, legal_moves)

        # Normalize model scores
        m_min = min(model_scores)
        m_max = max(model_scores)
        if m_max > m_min:
            norm_model_scores = [(s - m_min) / (m_max - m_min) for s in model_scores]
        else:
            norm_model_scores = [0.0 for _ in model_scores]

        # Normalize heuristic scores
        h_min = min(heuristic_scores)
        h_max = max(heuristic_scores)
        if h_max > h_min:
            norm_heuristic_scores = [(s - h_min) / (h_max - h_min) for s in heuristic_scores]
        else:
            norm_heuristic_scores = [0.0 for _ in heuristic_scores]

        best_idx = 0
        best_score = float("-inf")

        for i, mv in enumerate(filtered_moves):
            final_score = (
                self.alpha * norm_model_scores[i]
                + self.beta * norm_heuristic_scores[i]
            )

            if self.debug:
                print(
                    f"{mv.uci():6s} | "
                    f"model={model_scores[i]:9.4f} | "
                    f"norm_model={norm_model_scores[i]:6.3f} | "
                    f"heuristic={heuristic_scores[i]:9.3f} | "
                    f"norm_heur={norm_heuristic_scores[i]:6.3f} | "
                    f"final={final_score:8.4f}"
                )

            if final_score > best_score:
                best_score = final_score
                best_idx = i

        return filtered_moves[best_idx].uci()

    # ---------------------------
    # Main API
    # ---------------------------
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)