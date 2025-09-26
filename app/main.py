from tkinter import Tk
from app.gui import App
from app.controllers import DummyController

def main():
    root = Tk()
    root.title("HIT137 Assignment 3 — Tkinter + Hugging Face (Group)")
    root.geometry("940x600")
    controller = DummyController()  # will be replaced by real controller later
    App(root, controller=controller)
    root.mainloop()

if __name__ == "__main__":
    main()
