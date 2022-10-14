import tkinter

root = tkinter.Tk()

def toggle_video():
    # global show_video
    print("video toggle set to ", video_toggle_var.get())
    # show_video = video_toggle_var.get() == 1
video_toggle_var = tkinter.IntVar()
video_toggle = tkinter.Checkbutton(root, text="Show Video", variable=show_video, onvalue=1, offvalue=0, command=toggle_video)
video_toggle.grid(column=2, row=0)