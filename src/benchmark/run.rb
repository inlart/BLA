#!/usr/bin/ruby

require 'csv'
require 'json'

if(ENV["NPROC"])
    max_threads = Integer(ENV["NPROC"])
else
    max_threads = Integer(`cat /proc/cpuinfo | grep 'model name' | wc -l`)
end

if(ENV["BIN_DIR"])
    bin_folder = ENV["BIN_DIR"]
else
    bin_folder = "../../build/"
end

puts "NPROC: #{max_threads}"

extensions = Dir["#{bin_folder}/benchmark_*"]
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
for num_threads in 1..max_threads
    extensions.each do |extension|

        total_time = 0

        result_content = Array.new

        print "Running #{extension} with #{num_threads} thread(s)\n"

        ENV['NUM_WORKERS'] = num_threads.to_s
        ENV['OMP_NUM_THREADS'] = num_threads.to_s
        ENV['GOMP_CPU_AFFINITY'] = "0-3"

        ret = `#{bin_folder}#{extension} --benchmark_format=json`

        result = JSON.parse(ret)

        result["benchmarks"].each { | element |

            if element["time_unit"] != "ns"
                puts "time_unit not ns"
                exit
            end

            result_content.push(element["name"].split('/')[1])
            result_content.push(num_threads)
            result_content.push(element["name"].split('/')[0])
            result_content.push(element["real_time"])
            csv_content.push(result_content)
        }

    end

        CSV.open(filename, 'a+') do |csv_object|
            csv_content.each do |row_array|
                csv_object << row_array
            end
        end
        csv_content = Array.new
end
