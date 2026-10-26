using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

// A job object closes the local engine and its children when this app exits.
sealed class Launcher : Form {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr attr, string name);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int cls, IntPtr info, uint size);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [StructLayout(LayoutKind.Sequential)] struct BasicLimits { public long a,b; public uint flags; public UIntPtr c,d; public uint e; public UIntPtr f; public uint g,h; }
    [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong a,b,c,d,e,f; }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits { public BasicLimits basic; public IoCounters io; public UIntPtr a,b,c,d; }
    Process engine; IntPtr job; string url, ready, data; bool opened;
    Label status = new Label(); Button open = new Button(); System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
    readonly bool noBrowser=Array.IndexOf(Environment.GetCommandLineArgs(),"--no-browser")>=0;
    static string Quote(string s) { return "\"" + s + "\""; }
    Launcher() {
        Text="篮球集锦 · 本地预览版"; ClientSize=new Size(490,240); FormBorderStyle=FormBorderStyle.FixedDialog;
        MaximizeBox=false; StartPosition=FormStartPosition.CenterScreen; Font=new Font("Microsoft YaHei UI",10);
        status.SetBounds(24,22,440,70); status.Text="正在启动，请稍候…"; Controls.Add(status);
        open.Text="打开软件界面"; open.SetBounds(24,102,200,40); open.Enabled=false; open.Click+=(s,e)=>Open(); Controls.Add(open);
        Button folder=new Button(); folder.Text="打开本机任务文件夹"; folder.SetBounds(238,102,225,40); Controls.Add(folder);
        string[] args=Environment.GetCommandLineArgs(); int dataArg=Array.IndexOf(args,"--data-dir");
        data=dataArg>=0&&dataArg+1<args.Length?Path.GetFullPath(args[dataArg+1]):Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"BasketballHighlights"); Directory.CreateDirectory(data);
        folder.Click+=(s,e)=>Process.Start(new ProcessStartInfo(data){UseShellExecute=true});
        Label note=new Label(); note.SetBounds(24,160,440,60); note.Text="视频和识别均在本机处理。\n处理期间请保留此窗口，关闭它会停止正在运行的任务。"; Controls.Add(note);
        Shown+=(s,e)=>StartEngine(); FormClosing+=OnClosingApp;
        timer.Interval=300; timer.Tick+=(s,e)=>Check();
    }
    void StartEngine() {
        try {
            string root=AppDomain.CurrentDomain.BaseDirectory;
            ready=Path.Combine(data,"startup-"+Guid.NewGuid().ToString("N")+".txt");
            job=CreateJobObject(IntPtr.Zero,null);
            ExtendedLimits limits=new ExtendedLimits(); limits.basic.flags=0x2000;
            int size=Marshal.SizeOf(limits); IntPtr ptr=Marshal.AllocHGlobal(size);
            try { Marshal.StructureToPtr(limits,ptr,false); if(job==IntPtr.Zero||!SetInformationJobObject(job,9,ptr,(uint)size))throw new Exception("无法创建本地进程保护。"); }
            finally { Marshal.FreeHGlobal(ptr); }
            ProcessStartInfo info=new ProcessStartInfo(Path.Combine(root,"runtime","pythonw.exe"),Quote(Path.Combine(root,"app","scripts","run_windows_app.py"))+" --ready-file "+Quote(ready)+" --data-dir "+Quote(data));
            info.WorkingDirectory=root; info.UseShellExecute=false; info.CreateNoWindow=true;
            info.EnvironmentVariables.Remove("PYTHONHOME");info.EnvironmentVariables.Remove("PYTHONPATH");info.EnvironmentVariables["PYTHONNOUSERSITE"]="1";
            engine=Process.Start(info);
            if(!AssignProcessToJobObject(job,engine.Handle)){engine.Kill();throw new Exception("无法管理本地处理进程。");}
            timer.Start();
        } catch(Exception e) { status.Text="启动失败："+e.Message; }
    }
    void Check() {
        try {
            if(File.Exists(ready)) {
                if(url==null){url=File.ReadAllText(ready,Encoding.UTF8).Trim();Uri u=new Uri(url);if(u.Host!="127.0.0.1"||u.Scheme!="http")throw new Exception("本机地址无效。");}
                status.Text="软件已启动。\n在打开的本机页面中选择视频，确认进球后导出。";open.Enabled=true;
                if(!opened){opened=true;if(!noBrowser)Open();}
            }
            if(engine!=null&&engine.HasExited){timer.Stop();open.Enabled=false;status.Text=File.Exists(Path.ChangeExtension(ready,".error"))?File.ReadAllText(Path.ChangeExtension(ready,".error"),Encoding.UTF8):"处理服务已退出。请关闭窗口后重新启动软件。";}
        }catch(Exception e){timer.Stop();status.Text=e.Message;}
    }
    void Open(){if(url!=null)try{Process.Start(new ProcessStartInfo(url){UseShellExecute=true});}catch(Exception e){status.Text="请在浏览器打开："+url+"\n"+e.Message;}}
    void OnClosingApp(object sender,FormClosingEventArgs e){
        if(opened&&MessageBox.Show("关闭软件会停止正在进行的识别或导出。确定关闭？","关闭篮球集锦",MessageBoxButtons.YesNo,MessageBoxIcon.Question)!=DialogResult.Yes){e.Cancel=true;return;}
        timer.Stop();if(job!=IntPtr.Zero)CloseHandle(job);
        if(ready!=null){try{File.Delete(ready);File.Delete(Path.ChangeExtension(ready,".error"));}catch(IOException){}}
    }
    [STAThread] static void Main(){
        bool created;using(Mutex mutex=new Mutex(true,"Local\\BasketballHighlights",out created)){
            if(!created){MessageBox.Show("软件已经打开，请使用现有的篮球集锦窗口。","篮球集锦");return;}
            Application.EnableVisualStyles();Application.SetCompatibleTextRenderingDefault(false);Application.Run(new Launcher());
        }
    }
}
