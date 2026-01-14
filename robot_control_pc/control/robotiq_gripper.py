import time
import pymodbus

try:
    from pymodbus.client import ModbusSerialClient
except ImportError:
    from pymodbus.client.sync import ModbusSerialClient

class RobotiqGripperUSB:
    """Robotiq 2F-85/140 Gripper Control via USB (Modbus RTU)"""
    
    def __init__(self, portname="/dev/ttyUSB0", slave_id=9):
        self.portname = portname
        self.slave_id = slave_id 
        self.client = None
        self._error_count = 0
        self._is_v2 = pymodbus.__version__.startswith('2.')

    def connect(self):
        """Connects to the gripper"""
        if self._connect_core():
            time.sleep(1.0)
            return True
        return False

    def _connect_core(self):
        try:
            client_args = {
                'port': self.portname,
                'baudrate': 115200, 
                'stopbits': 1,
                'bytesize': 8,
                'parity': 'N',
                'timeout': 0.5 
            }
            if self._is_v2:
                client_args['method'] = 'rtu'

            self.client = ModbusSerialClient(**client_args)
            
            if self.client.connect():
                print(f"✅ Connected to {self.portname}")
                return True
            else:
                print(f"❌ Failed to open {self.portname}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def write_regs(self, addr, vals):
        try:
            if self._is_v2:
                return self.client.write_registers(addr, vals, unit=self.slave_id)
            else:
                return self.client.write_registers(address=addr, values=vals, slave=self.slave_id)
        except Exception as e:
            print(f"Write Error: {e}")
            return None
            
    def read_regs(self, addr, count):
        try:
            if self._is_v2:
                return self.client.read_holding_registers(addr, count, unit=self.slave_id)
            else:
                return self.client.read_holding_registers(address=addr, count=count, slave=self.slave_id)
        except Exception as e:
            class ErrorResult:
                def isError(self): return True
                def __str__(self): return str(e)
            return ErrorResult()

    def activate(self):
        """Activate the gripper"""
        status = self.get_status()
        if status:
            # If active and GTO set, skip (even if old fault exists)
            if status['activated'] == 1 and status['go_to'] == 1:
                print("✅ Gripper already active")
                return True

        print("⏳ Activating gripper...")
        try:
            # Reset
            self.write_regs(0x03E8, [0x0000, 0x0000, 0x0000])
            time.sleep(1.0)
            
            # Step 1: Activate only (no GTO)
            self.write_regs(0x03E8, [0x0100, 0x0000, 0xFFFF])
            
            # Wait for activation (gSTA == 3)
            for _ in range(20):
                time.sleep(0.5)
                s = self.get_status()
                if s and s['status'] == 3:
                    break
                if s and s['fault'] != 0 and s['fault'] != 13:
                    print(f"❌ Fault during activation: {s['fault']}")
                    return False
            else:
                print("❌ Activation timed out")
                return False
            
            # Step 2: Enable GTO
            self.write_regs(0x03E8, [0x0900, 0x0000, 0xFFFF])
            print("✅ Gripper Activated")
            return True
        except Exception as e:
            print(f"❌ Activation failed: {e}")
            return False 

    def close(self):
        """Close gripper fully"""
        try:
            self.write_regs(0x03E8, [0x0900, 0x00FF, 0xFF80])  # Speed 100%, Force 50%
            return True
        except Exception:
            return False

    def open(self):
        """Open gripper fully"""
        try:
            self.write_regs(0x03E8, [0x0900, 0x0000, 0xFF80])  # Speed 100%, Force 50%
            return True
        except Exception:
            return False

    def get_status(self):
        """Read gripper status registers"""
        try:
            result = self.read_regs(0x07D0, 3)
            if result.isError():
                self._error_count += 1
                if self._error_count > 5:
                    self.disconnect()
                    time.sleep(1.0)
                    self.connect()
                    self._error_count = 0
                return None
            
            self._error_count = 0
            status = result.registers

            status_byte = (status[0] >> 8) & 0xFF
            gACT = (status_byte >> 0) & 0x01
            gGTO = (status_byte >> 3) & 0x01
            gSTA = (status_byte >> 4) & 0x03
            gOBJ = (status_byte >> 6) & 0x03
            
            fault_byte = (status[1] >> 8) & 0xFF
            gFLT = fault_byte & 0x0F

            position = (status[2] >> 8) & 0xFF
            current = status[2] & 0xFF

            return {
                'activated': gACT,
                'go_to': gGTO,
                'status': gSTA,
                'object': gOBJ,
                'fault': gFLT,
                'position': position,
                'current': current
            }
        except Exception:
            self._error_count += 1
            return None
        
    def disconnect(self):
        if self.client:
            self.client.close()
