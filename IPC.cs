using Google.Protobuf;
using GTA;
using IPC;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IPC
{
    public enum FLAGS: int
    {
        REQUEST_GAME_STATE,
        REQUEST_ACTION,
        GAME_STATE_WRITTEN,
        ACTION_WRITTEN,
        RESET,
        IS_TRAINING
    }

    public class Flags
    {
        public const int NUMBER_OF_FLAGS = 5;
        public const string FLAGS_TAG = "flags.ipc";

        private readonly MemoryMappedFile ipc_flags_f;
        public readonly MemoryMappedViewStream flags;
        public Flags(int n = NUMBER_OF_FLAGS, string tag = FLAGS_TAG)
        {
            int size = -(n / -8);
            ipc_flags_f = MemoryMappedFile.OpenExisting(tag);
            flags = ipc_flags_f.CreateViewStream(0, size);
        }
        public void SetFlag(int idx, bool value)
        {
            if (value) SetFlag(idx, 1);
            else SetFlag(idx, 0);
        }
        private void SetFlag(int idx, uint value)
        {
            int pos = idx / 8;
            int offset = idx % 8;
            uint mask = ~((uint)1 << offset);
            flags.Seek(pos, 0);
            uint state = (uint)flags.ReadByte();
            flags.Seek(pos, 0);
            byte updated_state = (byte)((state & mask) | (value << offset));
            flags.WriteByte(updated_state);
            flags.Flush();
        }
        public void SetFlag(FLAGS idx, bool value)
        {
            SetFlag((int)idx, value);
        }

        public bool GetFlag(int idx)
        {
            int pos = idx / 8;
            int offset = idx % 8;
            int mask = 1 << offset;
            flags.Seek(pos, 0);
            int state = flags.ReadByte();
            return (state & mask) != 0;
        }
        public bool GetFlag(FLAGS idx)
        {
            return GetFlag((int)idx);
        }
        public void WaitUntil(int idx, bool value, Action fn)
        {
            while (GetFlag(idx) != value) fn();
        }
        public void WaitUntil(FLAGS idx, bool value, Action fn)
        {
            WaitUntil((int)idx, value, fn);
        }
        public void WaitUntil(int idx, bool value)
        {
            WaitUntil(idx, value, () => { });
        }
        public void WaitUntil(FLAGS idx, bool value)
        {
            WaitUntil((int)idx, value);
        }

        public void WaitUntil(IEnumerable<int> idxs, IEnumerable<bool> values, Action fn)
        {
            while (idxs.Zip(values, (idx, value) => GetFlag(idx) == value).All(x => x))
                fn();
        }
        public void WaitUntil(IEnumerable<FLAGS> idxs, IEnumerable<bool> values, Action fn)
        {
            WaitUntil(idxs.Cast<int>(), values, fn);
        }
        public void WaitUntil(IEnumerable<int> idxs, IEnumerable<bool> values)
        {
            WaitUntil(idxs, values, () => { });
        }
        public void WaitUntil(IEnumerable<FLAGS> idxs, IEnumerable<bool> values)
        {
            WaitUntil(idxs.Cast<int>(), values);
        }
    }

    public class Channel
    {

        private readonly MemoryMappedFile ipc_f;
        private readonly MemoryMappedViewStream ipc;
        private readonly int size;

        public Channel(string tag)
        {
            ipc_f = MemoryMappedFile.OpenExisting(tag);
            ipc = ipc_f.CreateViewStream();
        }

        public void Close()
        {
            ipc.Flush();
            ipc.Dispose();
            ipc_f.Dispose();
        }
        public void PushNbl(byte[] payload)
        {
            ipc.Seek(0, 0);
            ipc.Write(payload, 0, payload.Length);
            ipc.Flush();
        }
        public byte[] PopNbl()
        {
            int n = (int) ipc.Length;
            byte[] payload = new byte[n];
            ipc.Seek(0, 0);
            ipc.Read(payload, 0, n);
            return payload;
        }
    }

    public class MappedChannel<T>: Channel where T : IMessage, new()
    {

        public void PushNbl(T message)
        {
            PushNbl(message.ToByteArray());
        }
        public void PopNbl(ref T message)
        {
            byte[] message_bytes = base.PopNbl();
            message.MergeFrom(message_bytes);
        }

        public T PopNbl()
        {
            T msg = new T();
            byte[] message_bytes = base.PopNbl();
            msg.MergeFrom(message_bytes);
            return msg;
        }

        public MappedChannel(string tag) : base(tag) { }
    }

    class GameState : MappedChannel<Messages.GameState>
    {

        public Messages.GameState State;
        public Flags flags;

        public GameState(): base("game_state.ipc")
        {
            State = new Messages.GameState
            {
                CameraDirection = new Messages.Vector3 { },
                Velocity = new Messages.Vector3 { },
                Damage = 0
            };
            flags = new Flags();
        }
        public void Put(Messages.GameState state)
        {
            flags.WaitUntil(FLAGS.REQUEST_GAME_STATE, true);
            PushNbl(state);
            flags.SetFlag(FLAGS.GAME_STATE_WRITTEN, true);
        }
    }
}


    
//public void Debug()
//{
//    using (FileStream fsWrite = new FileStream("debug.txt", FileMode.OpenOrCreate, FileAccess.Write))
//    {
//        string[] debugstrs = new[] {
//            $"REQUEST_GAME_STATE: {GetFlag((int) FLAGS.REQUEST_GAME_STATE)}",
//            $"REQUEST_INPUT: {GetFlag((int) FLAGS.REQUEST_INPUT)}",
//            $"REQUEST_ACTION: {GetFlag((int) FLAGS.REQUEST_ACTION)}",
//            $"GAME_STATE_WRITTEN: {GetFlag((int) FLAGS.GAME_STATE_WRITTEN)}",
//            $"INPUT_WRITTEN: {GetFlag((int) FLAGS.INPUT_WRITTEN)}",
//            $"ACTION_WRITTEN: {GetFlag((int) FLAGS.ACTION_WRITTEN)}",
//            $"RESET: {GetFlag((int) FLAGS.RESET)}",
//            $"IS_TRAINING: {GetFlag((int) FLAGS.IS_TRAINING)}"
//        };
//        flags.Seek(0, 0);
//        debugstrs = debugstrs.Append($"{flags.ReadByte()}").ToArray();
//        string debugstr = string.Join("\n", debugstrs);
//        byte[] bytestr = System.Text.Encoding.UTF8.GetBytes(debugstr);
//        fsWrite.Write(bytestr, 0, bytestr.Length);
//    }
//}
