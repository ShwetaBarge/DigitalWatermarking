from watermarking import watermarking
from DE import get_de_values
import test

values = get_de_values()

Watermarking = watermarking(level=3, x=values[2])

# Gaussian noise attack

Watermarking.watermark()
test.add_gaussian_noise("watermarked_lena.jpg")
test.plot_image("lena.jpg", 'gaussian_noise_attack.jpg', 'Gaussian Attack')
test.print_data("Gaussian Noise Attack", "gaussian_noise_attack.jpg", "lena.jpg")

Watermarking.extracted(image_path="gaussian_noise_attack.jpg",
                       extracted_watermark_path="watermark_extracted_gaussian.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_gaussian.jpg")
print("For the Watermark image")
test.print_data("Gaussian Noise Attack", "watermark1.jpg", "watermark_extracted_gaussian.jpg")
test.plot_image("watermark1.jpg", 'watermark_extracted_gaussian.jpg', 'Gaussian Attack')

# Salt pepper noise

test.add_salt_pepper_noise("watermarked_lena.jpg", prob=0.01)
test.plot_image("watermarked_lena.jpg", "salt_pepper_noise_attack.jpg", 'Salt Pepper Attack')
test.print_data("Salt and Pepper Noise Attack", "salt_pepper_noise_attack.jpg", "watermarked_lena.jpg")

Watermarking.extracted(image_path="salt_pepper_noise_attack.jpg",
                       extracted_watermark_path="watermark_extracted_salt_pepper.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_salt_pepper.jpg")
print("For the Watermark image")
test.print_data("Salt and Pepper Attack", "watermark_extracted.jpg", "watermark_extracted_salt_pepper.jpg")
test.plot_image("watermark1.jpg", "watermark_extracted_salt_pepper.jpg", "Salt and Pepper Attack")

# Rotation attack

test.add_rotation_test(0, "watermarked_lena.jpg")
test.plot_image("watermarked_lena.jpg", "rotation_attack.jpg", 'Rotation Attack')
test.print_data("Rotation Attack", "rotation_attack.jpg", "watermarked_lena.jpg")

Watermarking.extracted(image_path="rotation_attack.jpg", extracted_watermark_path="watermark_extracted_rotation.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_rotation.jpg")
print("For the Watermark image")
test.print_data("Rotation Attack", "watermark_extracted.jpg", "watermark_extracted_rotation.jpg")
test.plot_image("watermark1.jpg", "watermark_extracted_rotation.jpg", "Rotation Attack ")

test.add_rotation_test(20, "watermarked_lena.jpg")
test.plot_image("watermarked_lena.jpg", "rotation_attack.jpg", 'Rotation Attack')
test.print_data("Rotation Attack", "rotation_attack.jpg", "watermarked_lena.jpg")

Watermarking.extracted(image_path="rotation_attack.jpg", extracted_watermark_path="watermark_extracted_rotation.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_rotation.jpg")
print("For the Watermark image")
test.print_data("Rotation Attack", "watermark_extracted.jpg", "watermark_extracted_rotation.jpg")
test.plot_image("watermark1.jpg", "watermark_extracted_rotation.jpg", "Rotation Attack ")

# Compression attack

test.compression_test("watermarked_lena.jpg")
#test.plot_image("watermarked_lena.jpg", "Compressed_watermarked_lena", 'Compression Attack')
test.print_data("Compression Attack", "Compressed_watermarked_lena.jpg", "watermarked_lena.jpg")

Watermarking.extracted(image_path="Compressed_watermarked_lena.jpg",
                       extracted_watermark_path="watermark_extracted_compression_attack.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_compression_attack.jpg")
print("For the Watermark image")
test.print_data("Compression Attack", "watermark_extracted.jpg", "watermark_extracted_compression_attack.jpg")
#test.plot_image("watermark1.jpg", "watermark_extracted_compression_attack.jpg", "Compression Attack")


# Cropping attack

test.rescale_test("watermarked_lena.jpg")
test.plot_image("watermarked_lena.jpg", "cropped_attack.jpg", 'Cropping Attack')
test.print_data("Crop Attack", "cropped_attack.jpg", "watermarked_lena.jpg")

Watermarking.extracted(image_path="cropped_attack.jpg", extracted_watermark_path="watermark_extracted_crop.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_crop.jpg")
print("For the Watermark image")
test.print_data("Cropping Attack", "watermark_extracted.jpg", "watermark_extracted_crop.jpg")
test.plot_image("watermark1.jpg", "watermark_extracted_crop.jpg", "Cropping Attack ")

# Poisson noise attack

test.add_poisson_noise("watermarked_lena.jpg")
test.plot_image("watermarked_lena.jpg", "poisson_noise_attack.jpg", 'Poisson Noise Attack')
test.print_data("Poisson Noise Attack", "poisson_noise_attack.jpg", "watermarked_lena.jpg")

Watermarking.extracted(image_path="poisson_noise_attack.jpg",
                       extracted_watermark_path="watermark_extracted_poisson_noise.jpg")
test.calculate_psnr_nc(img1="watermark1.jpg", img2="watermark_extracted_poisson_noise.jpg")
print("For the Watermark image")
test.print_data("Poisson Noise Attack", "watermark1.jpg", "watermark_extracted_poisson_noise.jpg")
test.plot_image("watermark1.jpg", "watermark_extracted.jpg", "Poisson Noise Attack ")



#testing at various degrees of rotation
test.testing("watermarked_lena.jpg")

test.rotate("watermarked_lena.jpg")
