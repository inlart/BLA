#!/usr/bin/ruby

require 'csv'

class Latex
    @@colors_count = 0
    @@colors = ["red", "blue", "yellow", "cyan", "gray", "green", "black", "purple", "peach"]
    @@api_colors = Hash.new
    @@plot_count = 1

    def self.add_api(api)
        if @@api_colors.has_key? api
            return
        end
        @@api_colors.store(api, @@colors[@@colors_count])
        puts
        @@colors_count += 1
    end

    def self.print_pgfplot(name, ylabel, data)

        puts "\\begin{tikzpicture}[scale=0.6]"
        puts "\\begin{axis}[xlabel=Threads, xtick distance=1, ylabel=#{ylabel}, xmin=1, ymin=0, legend pos=outer north east]"
        color = 0
        legend = Array.new
        data.each do |compiler, thread_tests|
            self.add_api(compiler)
            if thread_tests.size < 1
                next
            end
            used = false
            puts "\\addplot[color=#{@@api_colors[compiler]},mark=x] coordinates {"
            color += 1
            thread_tests.each do |num_threads, time|
                if Float(time) < 0
                    next
                end
                used = true
                puts "(#{num_threads}, #{Float(time).round(2)})"
            end
            puts "};"
            if used
                if compiler == "gcc"
                    legend.push("linear speed up")
                else
                    legend.push(compiler.gsub('_', ' ').gsub('/', ' ').split(' ')[-1])
                end
            end
        end

        puts "\\legend{#{legend.join(",")}}"

        puts "\\end{axis}"
        puts "\\end{tikzpicture}"
    end

    def self.start_plot()
        puts "\\begin{figure}"
    end

    def self.end_plot(name)
        puts "\\caption{#{name.gsub('_', ' ')}}"
        puts "\\label{fig:#{@@plot_count}}"
        puts "\\end{figure}"
        @@plot_count += 1
    end

end

class APIResult
    @name
    @results
    @program

    def initialize(name, program)
        @name = name
        @program = program
        @results = Hash.new
    end

    def get_name()
        return @name
    end

    def add_benchmark(num_threads, time)
        @results.store(num_threads, time)
    end

    def get_speed_up(num_threads = -1)
        if num_threads > 0
            return Float(@program.get_serial)/Float(@results[num_threads])
        end

        speed_up = Hash.new
        @results.each do |result, value|
            if Float(value) < 0
                next
            end
            speed_up.store(result, Float(@program.get_serial)/Float(value))
        end
        return speed_up
    end

    def get_efficiency(num_threads = -1)
        if num_threads > 0
            return (Float(@program.get_serial)/Float(@results[num_threads]))/Float(num_threads)
        end
        efficiency = Hash.new
        @results.each do |result, value|
            if Float(value) < 0
                next
            end
            efficiency.store(result, (Float(@program.get_serial)/Float(value))/Float(result))
        end
        return efficiency
    end

    def get_time(num_threads = -1)
        if num_threads > 0
            return @results[num_threads]
        end
        time = Hash.new
        @results.each do |result, value|
            if Float(value) < 0
                next
            end
            time.store(result, Float(value).round(2))
        end
        return time
    end

    def to_s()
        return @results.to_s
    end

end


class ProgramResult
    @name
    @serial
    @benchmarks
    @max_threads

    def initialize(name)
        @name = name
        @benchmarks = Hash.new
    end

    def get_benchmark(api)
        if @benchmarks.has_key?(api)
            return @benchmarks[api]
        else
            return nil
        end
    end

    def add_benchmark(compiler, num_threads, time)
        if not defined? @max_threads or Float(num_threads) > @max_threads
            @max_threads = Float(num_threads)
        end
        if Float(num_threads) == 1
            self.set_serial(time)
        end
        if not @benchmarks.has_key?(compiler)
            @benchmarks.store(compiler, APIResult.new(compiler, self))
        end
        @benchmarks[compiler].add_benchmark(num_threads, time)
    end

    def set_serial(serial)
        if not defined? @serial or (Float(serial) < Float(@serial) and Float(serial) > 0)
            @serial = serial
            return true
        end

        return false
    end

    def get_serial()
        return @serial
    end

    def get_speed_up()
        speed_up = Hash.new
        @benchmarks.each do | key, value |
            if value.get_name() == "gcc"
                next
            end
            speed_up.store(key, value.get_speed_up)
        end

        return speed_up
    end

    def get_efficiency()
        efficiency = Hash.new
        @benchmarks.each do | key, value |
            if value.get_name() == "gcc"
                next
            end
            efficiency.store(key, value.get_efficiency)
        end

        return efficiency
    end

    def get_time()
        time = Hash.new
        @benchmarks.each do | key, value |
            if value.get_name() == "gcc"
                next
            end
            time.store(key, value.get_time)
        end

        linear = Hash.new
        linear.store("1", Float(@serial))
        #puts @serial
        #exit
        for i in 2..@max_threads
            linear.store(String(i), Float(linear["1"]) / Float(i))
        end

        time.store("gcc", linear)

        return time
    end

    def to_s()
        return @benchmarks.to_s
    end

end

class ResultSet
    @results

    def initialize(file)
        @results = Hash.new

        CSV.foreach(file) do | row |
            program = row[0]

            num_threads = row[1]
            compiler = row[2]
            time = row[3]

            program = (compiler.split("_")[1]) + '\_' + program

            if not @results.has_key?(program)
                @results.store(program, ProgramResult.new(program))
            end

            @results[program].add_benchmark(compiler, num_threads, time)
        end
    end

    def print_latex()
        @results.each do |name, value|
            name = name.split('.')[0]
            Latex.start_plot
            Latex.print_pgfplot(name, "Runtime (ms)", value.get_time)
            Latex.print_pgfplot(name, "Speed up", value.get_speed_up)
            Latex.print_pgfplot(name, "Efficiency", value.get_efficiency)
            Latex.end_plot(name)
        end
    end
end

files = ARGV.length > 0 ? ARGV : ["results.csv"]

#TODO: add different line styles
style = []


puts "\\documentclass{article}

\\usepackage{pgfplots}

\\pgfplotsset{compat=1.14}

\\begin{document}

\\title{Allscale Benchmarks}"

files.each do |file|
    result = ResultSet.new(file)
    result.print_latex
end

puts "\\end{document}"
