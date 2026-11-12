library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity alu is
    generic (
        WIDTH : positive := 8
    );
    port (
        mode : in std_logic_vector(3 downto 0);
        a,b : in std_logic_vector (WIDTH - 1 downto 0);
        result : out std_logic_vector(WIDTH - 1 downto 0);
        is_negative : out std_logic;
        is_zero : out std_logic;
        overflow : out std_logic;
        carry : out std_logic
    );
end entity alu;

architecture behavioural of alu is
begin
    