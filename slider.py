import tkinter

# root = tkinter.Tk()
# root.title("Test slider")

root=tkinter.Tk()
root.geometry('400x600')

mAsk = tkinter.Scale(root, orient="horizontal", from_=1, to=16, label = "Mines", resolution = 1, sliderlength=25)
mAsk.pack()
root.mainloop()

root.mainloop()