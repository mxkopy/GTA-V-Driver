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
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


public class GrandTheftAutoReinforcementLearning : Script
{
    static readonly Vector3 AIRPORT = new Vector3(-1161.462f, -2584.786f, 13.505f);
    static readonly Vector3 HIGHWAY = new Vector3(-704.8778f, -2111.786f, 13.51563f);
    static readonly Vector3 HIGHWAY_DIRECTION = new Vector3(-0.7894784f, -0.6133158f, 0.02382357f);
    static GameState GameState = new GameState();
    static Flags Flags = GameState.flags;
    static Random rand = new Random();

    //public delegate void PresentCallback([MarshalAs(UnmanagedType.LPStruct)] IntPtr SwapChain);

    //[DllImport("dxinterop.asi", ExactSpelling = true, EntryPoint = "?getSwapChainPtr@@YAPEAXXZ")]
    //public static extern IntPtr GetSwapChainPtr();
    public GrandTheftAutoReinforcementLearning()
    {
        Tick += OnTick;
        KeyDown += OnKeyDown;

        Game.Player.Wanted.SetEveryoneIgnorePlayer(true);
        Game.Player.Wanted.SetPoliceIgnorePlayer(true);

        Flags.SetFlag(FLAGS.REQUEST_ACTION, true);

    }
    private void OnTick(object sender, EventArgs e)
    {
        Vehicle V = Game.Player.Character.CurrentVehicle;

        Game.Player.Wanted.SetWantedLevel(0, false);
        Game.Player.Wanted.ApplyWantedLevelChangeNow(false);

        GTA.Native.Hash.GET_FINAL_RENDERED_CAM_NEAR_CLIP

        if (V != null && Flags.GetFlag(FLAGS.IS_TRAINING))
        {
            Flags.WaitUntil(FLAGS.REQUEST_GAME_STATE, true);

            GameplayCamera.ForceRelativeHeadingAndPitch(0, 0, 0);
            GTA.Native.Function.Call(GTA.Native.Hash.FORCE_BONNET_CAMERA_RELATIVE_HEADING_AND_PITCH, 0, 0, 0);

            Vector3 CameraDirection = Vector3.Project(GameplayCamera.Direction, V.ForwardVector);
            GameState.State.CameraDirection.X = CameraDirection.X;
            GameState.State.CameraDirection.Y = CameraDirection.Y;
            GameState.State.CameraDirection.Z = CameraDirection.Z;

            Vector3 Velocity = Vector3.Project(V.Velocity, Vector3.Project(GameplayCamera.Direction, V.ForwardVector));
            GameState.State.Velocity.X = Velocity.X;
            GameState.State.Velocity.Y = Velocity.Y;
            GameState.State.Velocity.Z = Velocity.Z;

            GameState.State.Damage = (uint)(V.HasCollided ? 1 : 0);
            GameState.Put(GameState.State);

            Flags.SetFlag(FLAGS.REQUEST_GAME_STATE, false);

        }

        else if (V != null)
        {
            Reset();
            Wait(10);
        }
    }

    private void Reset()
    {

        Ped[] peds = World.GetAllPeds();
        Vehicle[] vehicles = World.GetAllVehicles(); 
        foreach (Ped ped in peds)
        {
            if (!Game.Player.Character.Equals(ped))
            {
                ped.Delete();
            }
        }
        foreach (Vehicle veh in vehicles)
        {
            veh.Delete();
        }
        Game.Player.Wanted.SetEveryoneIgnorePlayer(true);
        Game.Player.Wanted.SetPoliceIgnorePlayer(true);
        if (Game.Player.Character.CurrentVehicle != null)
        {
            Game.Player.Character.CurrentVehicle.Delete();
        }
        Game.Player.Character.Position = HIGHWAY;
        Vehicle vehicle = World.CreateVehicle(VehicleHash.EntityXF, Game.Player.Character.Position);
        Game.Player.Character.SetIntoVehicle(vehicle, VehicleSeat.Driver);
        vehicle.Heading = (float) rand.NextDouble() * 360f;

        GameplayCamera.ForceRelativeHeadingAndPitch(0, 0, 0);
        GTA.Native.Function.Call(GTA.Native.Hash.FORCE_BONNET_CAMERA_RELATIVE_HEADING_AND_PITCH, 0, 0, 0);

        //GameplayCamera.RelativeHeading = vehicle.Heading;
    }

    private void OnKeyDown(object sender, KeyEventArgs e)
    {
        if(e.KeyCode == Keys.Back)
        {
            Reset();
        }
    }
}