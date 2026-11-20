library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity ReferenceL1 is
    generic(
        programFile     : string := "program.hex";
        dataFile        : string := "data.hex";
        receiveFile     : string := "rx.hex";
        sendFile        : string := "tx.hex";
        resultFile      : string := "result.txt";
        dataBytes       : natural := 450432;
        lookupBase      : natural := 0;
        switches        : natural := 0
    );
end entity ReferenceL1;

architecture Behavioural of ReferenceL1 is
begin
    run : process
        type wordArray is array (0 to 4095) of natural;
        type byteArray is array (natural range <>) of integer;
        type byteArrayPointer is access byteArray;
        type registerArray is array (0 to 3) of natural;
        type pointerArray is array (0 to 2) of natural;
        file source         : text;
        file sink           : text;
        variable status     : file_open_status;
        variable row        : line;
        variable wordBits   : std_logic_vector(15 downto 0);
        variable byteBits   : std_logic_vector(7 downto 0);
        variable code       : wordArray := (others => 0);
        variable memory     : byteArrayPointer;
        variable received   : byteArrayPointer;
        variable receivedCount : natural := 0;
        variable receivedNext  : natural := 0;
        variable count      : natural := 0;
        variable reg        : registerArray := (others => 0);
        variable pointer    : pointerArray := (others => 0);
        variable lengthNow  : natural := 0;
        variable accumulator : integer := 0;
        variable product    : integer := 0;
        variable zero       : natural := 0;
        variable negative   : natural := 0;
        variable pc         : natural := 0;
        variable outByte    : natural := 0;
        variable cyclesLow  : natural := 0;
        variable cyclesHigh : natural := 0;
        variable halted     : boolean := false;
        variable waiting    : boolean := false;
        variable word       : natural;
        variable opcode     : natural;
        variable first      : natural;
        variable second     : natural;
        variable fn         : natural;
        variable low8       : natural;
        variable low12      : natural;
        variable which      : natural;
        variable step       : natural;
        variable value      : integer;
        variable shifted    : integer;
        variable products   : natural;
        variable lfsr       : unsigned(31 downto 0) := x"ACE1ACE1" xor to_unsigned(switches mod 65536, 32);
        variable feedback   : std_logic;
        variable sentBytes  : natural := 0;

        function signedByte(v : natural) return integer is
        begin
            if v > 127 then
                return v - 256;
            end if;
            return v;
        end function;

        function wrap24(v : integer) return integer is
            variable t : integer := v;
        begin
            if t >= 8388608 then
                t := t - 16777216;
            elsif t < -8388608 then
                t := t + 16777216;
            end if;
            return t;
        end function;

        function floorShift(v : integer; n : natural) return integer is
            variable t : integer := v;
        begin
            for g in 1 to n loop
                if t >= 0 then
                    t := t / 2;
                else
                    t := (t - 1) / 2;
                end if;
            end loop;
            return t;
        end function;

        function requantise(acc : integer; n : natural; relu : natural) return natural is
            variable t : integer := floorShift(acc, n);
        begin
            if relu = 1 and t < 0 then
                t := 0;
            end if;
            if t > 127 then
                t := 127;
            elsif t < -128 then
                t := -128;
            end if;
            if t < 0 then
                return t + 256;
            end if;
            return t;
        end function;

        procedure addCycles(n : natural) is
        begin
            cyclesLow := cyclesLow + n;
            if cyclesLow >= 1000000000 then
                cyclesLow := cyclesLow - 1000000000;
                cyclesHigh := cyclesHigh + 1;
            end if;
        end procedure;

        procedure setFlags(v : natural) is
        begin
            if v = 0 then
                zero := 1;
            else
                zero := 0;
            end if;
            negative := v / 128;
        end procedure;

        procedure stream(n : natural) is
            variable a : natural := pointer(0);
            variable b : natural := pointer(1);
            variable total : integer := accumulator;
        begin
            for g in 0 to n - 1 loop
                product := memory(a + g) * memory(b + g);
                total := wrap24(total + product);
            end loop;
            accumulator := total;
            pointer(0) := (a + n) mod 1048576;
            pointer(1) := (b + n) mod 1048576;
        end procedure;
    begin
        memory := new byteArray'(0 to dataBytes - 1 => 0);
        file_open(status, source, programFile, read_mode);
        assert status = open_ok report "cannot open " & programFile severity failure;
        while not endfile(source) loop
            readline(source, row);
            hread(row, wordBits);
            code(count) := to_integer(unsigned(wordBits));
            count := count + 1;
        end loop;
        file_close(source);
        count := 0;
        file_open(status, source, dataFile, read_mode);
        assert status = open_ok report "cannot open " & dataFile severity failure;
        while not endfile(source) loop
            readline(source, row);
            hread(row, byteBits);
            memory(count) := to_integer(signed(byteBits));
            count := count + 1;
        end loop;
        file_close(source);
        received := new byteArray'(0 to 65535 => 0);
        file_open(status, source, receiveFile, read_mode);
        if status = open_ok then
            while not endfile(source) loop
                readline(source, row);
                hread(row, byteBits);
                received(receivedCount) := to_integer(unsigned(byteBits));
                receivedCount := receivedCount + 1;
            end loop;
            file_close(source);
        end if;
        file_open(status, sink, sendFile, write_mode);

        while not halted and not waiting loop
            word := code(pc);
            pc := (pc + 1) mod 4096;
            opcode := word / 4096;
            first := (word / 1024) mod 4;
            second := (word / 256) mod 4;
            fn := word mod 16;
            low8 := word mod 256;
            low12 := word mod 4096;
            case opcode is
                when 0 =>
                    case fn is
                        when 1 => halted := true;
                        when 2 => reg(first) := switches mod 256;
                        when 3 => outByte := reg(first);
                        when 4 =>
                            value := memory(lookupBase + 256 * ((word / 16) mod 64) + reg(first));
                            if value < 0 then
                                value := value + 256;
                            end if;
                            reg(first) := value;
                            setFlags(value);
                        when 5 =>
                            hwrite(row, std_logic_vector(to_unsigned(reg(first), 8)));
                            writeline(sink, row);
                            sentBytes := sentBytes + 1;
                        when 6 =>
                            for g in 1 to 8 loop
                                feedback := lfsr(0) xor lfsr(10) xor lfsr(30) xor lfsr(31);
                                lfsr := feedback & lfsr(31 downto 1);
                            end loop;
                            reg(first) := to_integer(lfsr(31 downto 24));
                        when 7 =>
                            value := accumulator;
                            if value < 0 then
                                reg(first) := 24;
                            else
                                step := 0;
                                while value > 0 loop
                                    value := value / 2;
                                    step := step + 1;
                                end loop;
                                reg(first) := step;
                            end if;
                        when 8 =>
                            if receivedNext < receivedCount then
                                reg(first) := received(receivedNext);
                                receivedNext := receivedNext + 1;
                            else
                                waiting := true;
                                pc := (pc + 4095) mod 4096;
                            end if;
                        when others => null;
                    end case;
                when 1 => reg(first) := low8;
                when 2 =>
                    case fn is
                        when 0 => value := (reg(first) + reg(second)) mod 256;
                        when 1 => value := (reg(first) - reg(second)) mod 256;
                        when 2 => value := to_integer(to_unsigned(reg(first), 8) and to_unsigned(reg(second), 8));
                        when 3 => value := to_integer(to_unsigned(reg(first), 8) or to_unsigned(reg(second), 8));
                        when 4 => value := to_integer(to_unsigned(reg(first), 8) xor to_unsigned(reg(second), 8));
                        when 5 => value := reg(second);
                        when 6 => value := (reg(first) * 2) mod 256;
                        when 7 => value := reg(first) / 2;
                        when 8 => value := reg(first) / 2 + (reg(first) / 128) * 128;
                        when others => value := reg(first);
                    end case;
                    reg(first) := value;
                    setFlags(value);
                when 3 =>
                    if fn < 3 then
                        pointer(fn) := (pointer(fn) / 4096) * 4096 + (reg(first) mod 16) * 256 + reg(second);
                    elsif fn = 3 then
                        lengthNow := (reg(first) mod 16) * 256 + reg(second);
                    elsif fn < 7 then
                        pointer(fn - 4) := pointer(fn - 4) mod 4096 + reg(second) * 4096;
                    elsif fn >= 8 then
                        pointer((fn - 8) / 3) := pointer((fn - 8) mod 3);
                    end if;
                when 4 | 5 =>
                    if (word / 128) mod 2 = 1 then
                        which := 2;
                    else
                        which := (word / 512) mod 2;
                    end if;
                    step := (word / 256) mod 2;
                    if opcode = 4 then
                        value := memory(pointer(which));
                        if value < 0 then
                            value := value + 256;
                        end if;
                        reg(first) := value;
                    else
                        memory(pointer(which)) := signedByte(reg(first));
                    end if;
                    pointer(which) := (pointer(which) + step) mod 1048576;
                when 6 => product := signedByte(reg(first)) * signedByte(reg(second));
                when 7 =>
                    if fn = 0 then
                        accumulator := wrap24(accumulator + product);
                    else
                        value := reg(first) * 256 + reg(second);
                        if value >= 32768 then
                            value := value - 65536;
                        end if;
                        accumulator := value;
                    end if;
                when 8 =>
                    if (word / 256) mod 2 = 1 then
                        products := lengthNow;
                        addCycles(products);
                    else
                        products := low8;
                        addCycles(products);
                        if products = 0 then
                            products := 1;
                        end if;
                    end if;
                    stream(products);
                when 9 =>
                    if (word / 256) mod 2 = 1 then
                        reg(first) := requantise(accumulator, reg(word mod 4) mod 32, (word / 512) mod 2);
                    else
                        reg(first) := requantise(accumulator, word mod 32, (word / 512) mod 2);
                    end if;
                when 10 => pointer(0) := (pointer(0) / 4096) * 4096 + low12;
                when 11 => pointer(1) := (pointer(1) / 4096) * 4096 + low12;
                when 12 => pc := low12;
                when 13 =>
                    if zero = 1 then
                        pc := low12;
                    end if;
                when 14 =>
                    if zero = 0 then
                        pc := low12;
                    end if;
                when others =>
                    reg(3) := (reg(3) + 255) mod 256;
                    if reg(3) /= 0 then
                        pc := low12;
                    end if;
            end case;
            if not waiting then
                addCycles(4);
            end if;
        end loop;
        file_close(sink);

        file_open(status, sink, resultFile, write_mode);
        write(row, string'("cycles_high ") & integer'image(cyclesHigh));
        writeline(sink, row);
        write(row, string'("cycles_low ") & integer'image(cyclesLow));
        writeline(sink, row);
        write(row, string'("out ") & integer'image(outByte));
        writeline(sink, row);
        write(row, string'("registers ") & integer'image(reg(0)) & " " & integer'image(reg(1)) & " " & integer'image(reg(2)) & " " & integer'image(reg(3)));
        writeline(sink, row);
        write(row, string'("pointers ") & integer'image(pointer(0)) & " " & integer'image(pointer(1)) & " " & integer'image(pointer(2)));
        writeline(sink, row);
        write(row, string'("len ") & integer'image(lengthNow));
        writeline(sink, row);
        write(row, string'("acc ") & integer'image(accumulator));
        writeline(sink, row);
        write(row, string'("sent ") & integer'image(sentBytes));
        writeline(sink, row);
        if halted then
            write(row, string'("ended halted"));
        else
            write(row, string'("ended waiting"));
        end if;
        writeline(sink, row);
        file_close(sink);
        wait;
    end process run;
end architecture Behavioural;
