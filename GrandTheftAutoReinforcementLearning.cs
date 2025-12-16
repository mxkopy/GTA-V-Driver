using GTA;
using GTA.Math;
using GTA.NaturalMotion;
using IPC;
using SharpDX.Direct3D11;
using SharpDX.Mathematics.Interop;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


public class GrandTheftAutoReinforcementLearning : Script
{
    static readonly Vector3 AIRPORT = new Vector3(-1161.462f, -2584.786f, 13.505f);
    static GameState GameState = new GameState();
    static Flags flags = GameState.flags;

    //public delegate void PresentCallback([MarshalAs(UnmanagedType.LPStruct)] IntPtr SwapChain);

    //[DllImport("dxinterop.asi", ExactSpelling = true, EntryPoint = "?getSwapChainPtr@@YAPEAXXZ")]
    //public static extern IntPtr GetSwapChainPtr();

    public GrandTheftAutoReinforcementLearning()
    {
        Tick += OnTick;
        KeyDown += OnKeyDown;

        Game.Player.Wanted.SetEveryoneIgnorePlayer(true);
        Game.Player.Wanted.SetPoliceIgnorePlayer(true);

    }
    private void OnTick(object sender, EventArgs e)
    {
        Vehicle V = Game.Player.Character.CurrentVehicle;

        Game.Player.Wanted.SetWantedLevel(0, false);
        Game.Player.Wanted.ApplyWantedLevelChangeNow(false);


        if (V != null)
        {
            if (flags.GetFlag((int)FLAGS.IS_TRAINING))
            {
                Vector3 CameraDirection = Vector3.Project(GameplayCamera.Direction, V.ForwardVector);
                GameState.State.CameraDirection.X = CameraDirection.X;
                GameState.State.CameraDirection.Y = CameraDirection.Y;
                GameState.State.CameraDirection.Z = CameraDirection.Z;
                Vector3 Velocity = Vector3.Project(V.Velocity, Vector3.Project(GameplayCamera.Direction, V.ForwardVector));
                GameState.State.Velocity.X = Velocity.X;
                GameState.State.Velocity.Y = Velocity.Y;
                GameState.State.Velocity.Z = Velocity.Z;

                GameState.State.Damage = (uint)(V.MaxHealth - V.Health);

                GameState.Put(GameState.State);

                //GTA.UI.Notification.PostTicker($"{GetSwapChainPtr()}", true);

                while (flags.GetFlag((int)FLAGS.GAME_STATE_WRITTEN) && flags.GetFlag((int)FLAGS.IS_TRAINING)) ;
            }
            else
            {
                Reset();
                //Yield();
                Wait(10);
            }
        }
    }

    private void Reset()
    {
        Game.Player.Wanted.SetEveryoneIgnorePlayer(true);
        Game.Player.Wanted.SetPoliceIgnorePlayer(true);
        if (Game.Player.Character.CurrentVehicle != null)
        {
            Game.Player.Character.CurrentVehicle.Delete();
        }
        Game.Player.Character.Position = AIRPORT;
        Vehicle vehicle = World.CreateVehicle(VehicleHash.EntityXF, Game.Player.Character.Position);
        Game.Player.Character.SetIntoVehicle(vehicle, VehicleSeat.Driver);
    }

    private void OnKeyDown(object sender, KeyEventArgs e)
    {
        if(e.KeyCode == Keys.Back)
        {
            Reset();
        }
    }
}