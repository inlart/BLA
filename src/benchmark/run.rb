#!/usr/bin/ruby

require 'csv'

if(ENV["NPROC"])
    max_threads = Integer(ENV["NPROC"])
else
    max_threads = Integer(`cat /proc/cpuinfo | grep 'model name' | wc -l`)
end

if(ENV["NREP"])
    num_executions = Integer(ENV["NREP"])
else
    num_executions = 1
end

if(ENV["BIN_DIR"])
    bin_folder = ENV["BIN_DIR"]
else
    bin_folder = "../../build/"
end

puts "NPROC: #{max_threads}"
puts "NREP: #{num_executions}"

inputs = [128, 256, 512, 1024, 2048, 4096]

extensions = Dir["#{bin_folder}/bench_*"]
extensions.map! { |item| File.basename item}
extensions.sort!

if ARGV[0]
    filename = ARGV[0]
else
    filename = "results.csv"
end

if File.exists? filename
    File.delete(filename)
end

csv_content = Array.new
inputs.each do |input|
    for num_threads in 1..max_threads
        extensions.each do |extension|

            total_time = 0

            result_content = Array.new
            result_content.push(input)
            result_content.push(num_threads)
            result_content.push(extension)

            #run the binary multiple times to minimize measuring errors
            print "Running #{extension} with #{num_threads} thread(s) and matrix size #{input}\n"
            for i in 1..num_executions
                ENV['NUM_WORKERS'] = num_threads.to_s
                ENV['OMP_NUM_THREADS'] = num_threads.to_s
                ENV['GOMP_CPU_AFFINITY'] = "0-3"

                time = 0

                ret = `#{bin_folder}#{extension} #{input} raw`
                puts ret
                values = ret.split(' ').map(&:to_f)
                time = values.inject{ |sum, el| sum + el }.to_f / values.size

                total_time += time
            end

            #calculate the average runtime and write
            if(total_time < 0)
                average_time = -1
            else
                average_time = total_time / num_executions
            end

            puts "Time: #{average_time}"

            result_content.push(average_time)
            csv_content.push(result_content)
        end

            CSV.open(filename, 'a+') do |csv_object|
                csv_content.each do |row_array|
                    csv_object << row_array
                end
            end
            csv_content = Array.new
    end
end
