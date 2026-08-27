import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
from typing import Dict, Any, Optional

from python_mini_ai import config
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.knowledge.knowledge_base import KnowledgeBase
from python_mini_ai.memory.memory_manager import MemoryManager
from python_mini_ai.ai.neural_network import NeuralNetwork
from python_mini_ai.ai.trainer import ModelTrainer
from python_mini_ai.ai.inference import InferenceEngine
from python_mini_ai.ai.intent_classifier import IntentClassifier
from python_mini_ai.ai.response_generator import ResponseGenerator
from python_mini_ai.ai.decision_engine import DecisionEngine
from python_mini_ai.training.learning_system import LearningSystem

logger = logging.getLogger(__name__)

class ChatWindow:
    """Modern Dark Mode Desktop GUI for Python Mini AI using Tkinter."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Python Mini AI - Local Intelligence Engine")
        self.root.geometry("960x680")
        self.root.minsize(800, 550)
        self.root.configure(bg="#1e1e2e")

        # Color palette
        self.bg_color = "#1e1e2e"
        self.panel_color = "#25263a"
        self.chat_bg = "#181825"
        self.user_bubble_bg = "#313244"
        self.ai_bubble_bg = "#45475a"
        self.accent_color = "#89b4fa"
        self.text_color = "#cdd6f4"
        self.subtext_color = "#a6adc8"
        self.success_color = "#a6e3a1"
        self.warning_color = "#f9e2af"
        self.danger_color = "#f38ba8"

        self.last_user_query = ""
        self.last_ai_response = ""
        self.last_intent = ""

        self._init_ai_components()
        self._build_ui()

    def _init_ai_components(self):
        """Initializes data managers, neural network model, and inference components."""
        self.dataset_manager = DatasetManager(config.TRAINING_DATA_PATH)
        self.knowledge_base = KnowledgeBase(config.KNOWLEDGE_PATH)
        self.memory_manager = MemoryManager(config.DB_PATH, max_short_term=config.SHORT_TERM_MEMORY_SIZE)

        # Load or train Neural Network
        self.model = NeuralNetwork(layer_sizes=[10, 64, 32, 5])
        if config.MODEL_PATH.exists():
            self.model.load(config.MODEL_PATH)
        else:
            trainer = ModelTrainer(self.dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
            self.model, _ = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
            self.model.save(config.MODEL_PATH)

        self.inference_engine = InferenceEngine(self.model, self.dataset_manager)
        self.intent_classifier = IntentClassifier(self.inference_engine)
        self.response_generator = ResponseGenerator(self.dataset_manager)
        self.decision_engine = DecisionEngine(
            intent_classifier=self.intent_classifier,
            response_generator=self.response_generator,
            knowledge_base=self.knowledge_base,
            memory_manager=self.memory_manager
        )
        self.learning_system = LearningSystem(
            dataset_manager=self.dataset_manager,
            memory_manager=self.memory_manager,
            knowledge_base=self.knowledge_base,
            model=self.model
        )

    def _build_ui(self):
        """Builds GUI layout with Chat area, Side Panel (Debug/Confidence), and Bottom Input bar."""
        # Main Layout: Left = Chat, Right = Debug/Status Panel
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Column (Chat Area)
        left_frame = tk.Frame(main_container, bg=self.bg_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Header Bar
        header = tk.Frame(left_frame, bg=self.panel_color, height=40)
        header.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            header,
            text=" Python Mini AI",
            font=("Helvetica", 14, "bold"),
            bg=self.panel_color,
            fg=self.accent_color
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.status_label = tk.Label(
            header,
            text="● ONLINE (Local Engine)",
            font=("Helvetica", 10, "bold"),
            bg=self.panel_color,
            fg=self.success_color
        )
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Chat History Log
        self.chat_display = scrolledtext.ScrolledText(
            left_frame,
            wrap=tk.WORD,
            bg=self.chat_bg,
            fg=self.text_color,
            font=("Consolas", 11),
            insertbackground=self.text_color,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)

        # Bottom Input Area
        input_frame = tk.Frame(left_frame, bg=self.panel_color)
        input_frame.pack(fill=tk.X)

        self.entry_field = tk.Entry(
            input_frame,
            bg=self.user_bubble_bg,
            fg=self.text_color,
            font=("Helvetica", 12),
            insertbackground=self.text_color,
            relief=tk.FLAT
        )
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        self.entry_field.bind("<Return>", lambda event: self.send_message())

        send_btn = tk.Button(
            input_frame,
            text="Senden",
            font=("Helvetica", 11, "bold"),
            bg=self.accent_color,
            fg="#11111b",
            activebackground="#b4befe",
            relief=tk.FLAT,
            padx=15,
            command=self.send_message
        )
        send_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        # Right Column (Debug & Controls Panel)
        right_frame = tk.Frame(main_container, bg=self.panel_color, width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Debug Title
        panel_title = tk.Label(
            right_frame,
            text="AI Debug & Status",
            font=("Helvetica", 12, "bold"),
            bg=self.panel_color,
            fg=self.text_color
        )
        panel_title.pack(anchor="w", padx=10, pady=10)

        # Confidence Gauge Display
        conf_frame = tk.LabelFrame(
            right_frame,
            text=" Confidence Score ",
            font=("Helvetica", 10, "bold"),
            bg=self.panel_color,
            fg=self.subtext_color,
            padx=10,
            pady=10
        )
        conf_frame.pack(fill=tk.X, padx=10, pady=5)

        self.conf_value_label = tk.Label(
            conf_frame,
            text="0.00 %",
            font=("Helvetica", 18, "bold"),
            bg=self.panel_color,
            fg=self.accent_color
        )
        self.conf_value_label.pack()

        self.conf_progress = ttk.Progressbar(conf_frame, orient="horizontal", mode="determinate")
        self.conf_progress.pack(fill=tk.X, pady=(5, 0))

        # Intelligence Details Box
        details_frame = tk.LabelFrame(
            right_frame,
            text=" AI Engine Details ",
            font=("Helvetica", 10, "bold"),
            bg=self.panel_color,
            fg=self.subtext_color,
            padx=10,
            pady=10
        )
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.debug_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            bg=self.chat_bg,
            fg=self.subtext_color,
            font=("Consolas", 9),
            relief=tk.FLAT,
            height=12
        )
        self.debug_text.pack(fill=tk.BOTH, expand=True)
        self.debug_text.config(state=tk.DISABLED)

        # Feedback Buttons (+ / -)
        feedback_frame = tk.Frame(right_frame, bg=self.panel_color)
        feedback_frame.pack(fill=tk.X, padx=10, pady=5)

        btn_pos = tk.Button(
            feedback_frame,
            text="👍 Positiv",
            font=("Helvetica", 9, "bold"),
            bg=self.success_color,
            fg="#11111b",
            relief=tk.FLAT,
            command=lambda: self.handle_feedback(True)
        )
        btn_pos.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_neg = tk.Button(
            feedback_frame,
            text="👎 Negativ",
            font=("Helvetica", 9, "bold"),
            bg=self.danger_color,
            fg="#11111b",
            relief=tk.FLAT,
            command=lambda: self.handle_feedback(False)
        )
        btn_neg.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Action Buttons (Train / Clear Memory)
        action_frame = tk.Frame(right_frame, bg=self.panel_color)
        action_frame.pack(fill=tk.X, padx=10, pady=10)

        btn_teach = tk.Button(
            action_frame,
            text="🎓 KI Trainieren",
            font=("Helvetica", 10, "bold"),
            bg=self.accent_color,
            fg="#11111b",
            relief=tk.FLAT,
            command=self.open_training_dialog
        )
        btn_teach.pack(fill=tk.X, pady=(0, 5))

        btn_clear = tk.Button(
            action_frame,
            text="🗑 Chat Leeren",
            font=("Helvetica", 10),
            bg=self.user_bubble_bg,
            fg=self.text_color,
            relief=tk.FLAT,
            command=self.clear_chat
        )
        btn_clear.pack(fill=tk.X)

        # Welcome message
        self._append_message("AI", "Hallo! Ich bin Python Mini AI. Wie kann ich dir heute helfen?")

    def _append_message(self, sender: str, text: str):
        """Appends formatted message turn to the scrolled text widget."""
        self.chat_display.config(state=tk.NORMAL)
        if sender == "User":
            self.chat_display.insert(tk.END, f"\nUser: {text}\n", "user_tag")
        else:
            self.chat_display.insert(tk.END, f"\nAI: {text}\n", "ai_tag")
        self.chat_display.tag_config("user_tag", foreground=self.accent_color, font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("ai_tag", foreground=self.success_color, font=("Consolas", 11))
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def send_message(self):
        """Process user message send action."""
        user_text = self.entry_field.get().strip()
        if not user_text:
            return

        self.entry_field.delete(0, tk.END)
        self._append_message("User", user_text)

        result = self.decision_engine.process_query(user_text)
        ai_response = result["response"]
        confidence = result["confidence"]
        debug_info = result["debug"]

        self._append_message("AI", ai_response)

        self.last_user_query = user_text
        self.last_ai_response = ai_response
        self.last_intent = debug_info.get("detected_intent", "")

        self._update_debug_panel(confidence, debug_info)

    def _update_debug_panel(self, confidence: float, debug_info: Dict[str, Any]):
        """Updates the confidence progress gauge and debug text view."""
        pct = confidence * 100
        self.conf_value_label.config(text=f"{pct:.1f} %")
        self.conf_progress["value"] = pct

        self.debug_text.config(state=tk.NORMAL)
        self.debug_text.delete("1.0", tk.END)

        info_str = (
            f"Input: {debug_info.get('input', '')}\n"
            f"Source: {debug_info.get('response_source', '')}\n"
            f"Intent: {debug_info.get('detected_intent', '')}\n"
            f"NN Confidence: {debug_info.get('nn_confidence', 0.0):.2f}\n"
            f"KB Similarity: {debug_info.get('kb_score', 0.0):.2f}\n"
            f"Final Confidence: {confidence:.2f}\n"
        )
        self.debug_text.insert(tk.END, info_str)
        self.debug_text.config(state=tk.DISABLED)

    def handle_feedback(self, is_positive: bool):
        """Processes positive or negative user feedback."""
        if not self.last_user_query:
            messagebox.showinfo("Feedback", "Noch keine Nachricht zum Bewerten vorhanden.")
            return

        res = self.learning_system.record_feedback(
            user_query=self.last_user_query,
            ai_response=self.last_ai_response,
            is_positive=is_positive,
            tag=self.last_intent
        )
        messagebox.showinfo("Feedback erhalten", res["message"])

    def open_training_dialog(self):
        """Opens a modal dialog to directly teach new Q&A pairs."""
        dialog = tk.Toplevel(self.root)
        dialog.title("KI Neue Frage beibringen")
        dialog.geometry("450x300")
        dialog.configure(bg=self.panel_color)

        tk.Label(
            dialog,
            text="Frage / Muster:",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Helvetica", 10, "bold")
        ).pack(anchor="w", padx=15, pady=(15, 2))

        q_entry = tk.Entry(dialog, bg=self.user_bubble_bg, fg=self.text_color, font=("Helvetica", 11))
        q_entry.pack(fill=tk.X, padx=15, pady=(0, 10))

        tk.Label(
            dialog,
            text="Gewünschte Antwort:",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Helvetica", 10, "bold")
        ).pack(anchor="w", padx=15, pady=(5, 2))

        a_entry = tk.Entry(dialog, bg=self.user_bubble_bg, fg=self.text_color, font=("Helvetica", 11))
        a_entry.pack(fill=tk.X, padx=15, pady=(0, 15))

        def submit_teaching():
            q = q_entry.get().strip()
            a = a_entry.get().strip()
            if not q or not a:
                messagebox.showwarning("Fehler", "Bitte sowohl Frage als auch Antwort ausfüllen.")
                return

            self.status_label.config(text="● TRAINING...", fg=self.warning_color)
            dialog.destroy()

            def train_task():
                ok, msg = self.learning_system.teach_new_qa(q, a)
                self.root.after(0, lambda: self._on_training_complete(ok, msg))

            threading.Thread(target=train_task, daemon=True).start()

        tk.Button(
            dialog,
            text="Lernen & Trainieren",
            font=("Helvetica", 11, "bold"),
            bg=self.accent_color,
            fg="#11111b",
            relief=tk.FLAT,
            command=submit_teaching
        ).pack(pady=10)

    def _on_training_complete(self, ok: bool, msg: str):
        """Callback when background re-training finishes."""
        self.status_label.config(text="● ONLINE (Local Engine)", fg=self.success_color)
        if ok:
            messagebox.showinfo("Lernfortschritt", msg)
        else:
            messagebox.showerror("Fehler", msg)

    def clear_chat(self):
        """Clears GUI chat view and short term memory queue."""
        self.memory_manager.clear_short_term()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._append_message("AI", "Chatverlauf geleert. Wie kann ich dir weiterhelfen?")


def launch_gui():
    """Launches Tkinter mainloop or raises error if display unavailable."""
    root = tk.Tk()
    app = ChatWindow(root)
    root.mainloop()
