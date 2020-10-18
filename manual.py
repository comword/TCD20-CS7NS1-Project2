import tkinter as tk
from PIL import ImageTk, Image
from pathlib import Path
import argparse

class ManualMain:
    def __init__(self, args):
        self.path = Path(args.path)
        self.list = args.list
        self.list = self.load_file()
        self.cur_num = 0
        self.output = args.output
        self.build_window()
        self.refresh()

    def func_btn_next(self):
        if self.cur_num < len(self.list):
            self.update()
            self.cur_num += 1
            self.refresh()
            

    def func_btn_prev(self):
        if self.cur_num > 0:
            self.update()
            self.cur_num -= 1
            self.refresh()

    def func_btn_save(self):
        with open(self.output, "w") as f:
            for l in self.list:
                f.write("%s,%s\n" % (l[0],l[1]))

    def build_window(self):
        window = tk.Tk()
        window.title('Window')
        window.geometry('500x300')
        canvas = tk.Canvas(window, width=400, height=135)
        canvas.pack(side='top')
        btn_prev = tk.Button(window, text='Prev', command=self.func_btn_prev)
        btn_prev.place(x=120, y=240)
        btn_next = tk.Button(window, text='Next', command=self.func_btn_next)
        btn_next.place(x=200, y=240)
        btn_save = tk.Button(window, text='Save', command=self.func_btn_save)
        btn_save.place(x=280, y=240)

        var_code = tk.StringVar()
        entry_code = tk.Entry(window, textvariable=var_code)
        entry_code.place(x=130, y=110)

        self.window = window
        self.canvas = canvas
        self.var_code = var_code

    def update(self):
        new_code = self.var_code.get()
        fname, old_code = self.list[self.cur_num]
        self.list[self.cur_num] = (fname, new_code)

    def refresh(self):
        self.canvas.delete("all")
        self.image_file = tk.PhotoImage(file=self.path / self.list[self.cur_num][0])
        self.image = self.canvas.create_image(200, 0, anchor='n', image=self.image_file)
        self.var_code.set(self.list[self.cur_num][1])
        self.window.update()
    
    def load_file(self):
        l = list()
        with open(self.list, "r") as f:
            for line in f:
                line = line.rstrip('\n')
                fname, code = line.split(",")
                l.append((fname, code))
        return l

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--path', type=str, required=True,
                      help='path to image to be processed')
    args.add_argument('-l', '--list', type=str, required=True,
                      help='wrong list file')
    args.add_argument('-o', '--output', type=str, required=True,
                      help='file to output')
    arg_parsed = args.parse_args()
    m = ManualMain(arg_parsed)
    m.run()
